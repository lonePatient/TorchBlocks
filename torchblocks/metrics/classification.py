import torch
import logging
from tqdm import tqdm
import numpy as np
from torchblocks.metrics.base import Metric
from torchblocks.utils.tensor import tensor_to_numpy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)


class Accuracy(Metric):
    '''
    Computes accuracy. Works with binary, multiclass, and multilabel data.
    Accepts logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Args:
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
    Example:
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)
    '''

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def reset(self):
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

    def update(self, input, target):
        if not (len(input.shape) == len(target.shape) or len(input.shape) == len(target.shape) + 1):
            raise ValueError(
                "preds and target must have same number of dimensions, or one additional dimension for preds"
            )
        if len(input.shape) == len(target.shape) + 1:
            preds = torch.argmax(input, dim=1)
        elif len(input.shape) == len(target.shape) and input.dtype == torch.float:
            preds = (input >= self.threshold).long()
        else:
            preds = input

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def value(self):
        return (self.correct.float() / self.total).item()

    def name(self):
        return 'acc'


class MattewsCorrcoef(Metric):
    '''
    Matthews Correlation Coefficient
    '''

    def update(self, input, target):
        self.preds = torch.argmax(input, dim=1)
        self.labels = target

    def value(self):
        return matthews_corrcoef(tensor_to_numpy(self.labels), tensor_to_numpy(self.preds))

    def name(self):
        return 'mcc'


class AUC(Metric):
    '''
    Area Under Curve
    '''

    def __init__(self, task_type='binary', average='binary'):
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average

    def update(self, input, target):
        if self.task_type == 'binary':
            self.y_prob = tensor_to_numpy(input.sigmoid().data)
        else:
            self.y_prob = tensor_to_numpy(input.softmax(-1).data)
        self.y_true = tensor_to_numpy(target)

    def value(self):
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        return auc

    def name(self):
        return 'auc'


class F1Score(Metric):
    '''
    F1 Score
    '''

    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='binary', search_thresh=False):

        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self, y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们队Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def update(self, input, target):

        self.y_true = tensor_to_numpy(target)
        if self.normalizate and self.task_type == 'binary':
            y_prob = tensor_to_numpy(input.sigmoid().data)
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = tensor_to_numpy(input.softmax(-1).data)
        else:
            y_prob = tensor_to_numpy(input)

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
            else:
                thresh, f1 = self.thresh_search(y_prob=y_prob)
                logger.info(f"Best Thresh: {thresh:.4f} - F1 Score: {f1:.4f}")
        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def value(self):
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'


class ClassReport(Metric):
    '''
    classification report
    '''

    def __init__(self, target_names=None):
        self.target_names = target_names

    def update(self, input, target):
        _, y_pred = torch.max(input, 1)
        self.y_pred = tensor_to_numpy(y_pred)
        self.y_true = tensor_to_numpy(target)

    def value(self):
        score = classification_report(y_true=self.y_true, y_pred=self.y_pred, target_names=self.target_names)
        logger.info(f"\n\n classification report: {score}")

    def name(self):
        return "class_report"


class MultiLabelReport(Metric):
    '''
    multi label report
    '''

    def __init__(self, id2label=None):

        self.id2label = id2label

    def update(self, input, target):
        self.y_prob = tensor_to_numpy(input.sigmoid().data)
        self.y_true = tensor_to_numpy(target)

    def value(self):
        for i, label in self.id2label.items():
            try:
                auc = roc_auc_score(y_score=self.y_prob[:, i], y_true=self.y_true[:, i])
            except Exception as e:
                auc = 0.000
                logger.warning(f"Only one class present in label:{label} ")
            logger.info(f"  label:{label} - auc: {auc:.4f}")

    def name(self):
        return "multilabel_report"
