import torch
import logging
from tqdm import tqdm
import numpy as np
from .base import Metric
from ..utils.tensor import tensor_to_numpy

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


class Accuracy(Metric):
    '''
    Accuracy
    '''

    def __init__(self):
        super().__init__()

    def update(self, input, target):
        if input.dim() == 1:
            self.preds = input.numpy()
        else:
            self.preds = torch.argmax(input, dim=1).numpy()
        self.labels = target.numpy()

    def value(self):
        return simple_accuracy(self.preds, self.labels)

    def name(self):
        return 'acc'


class MattewsCorrcoef(Metric):
    '''
    Matthews Correlation Coefficient
    '''

    def __init__(self):
        super().__init__()

    def update(self, input, target):
        self.preds = torch.argmax(input, dim=1).numpy()
        self.labels = target.numpy()

    def value(self):
        return matthews_corrcoef(self.labels, self.preds)

    def name(self):
        return 'mcc'


class AccuracyThresh(Metric):
    '''
    Accuracy with thresh
    '''

    def __init__(self, thresh=0.5):
        super(AccuracyThresh, self).__init__()
        self.thresh = thresh

    def update(self, input, target):
        self.y_pred = input.sigmoid()
        self.y_true = target

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean(tensor_to_numpy(((self.y_pred > self.thresh) == self.y_true.byte()).float()), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'acc'


class AUC(Metric):
    '''
    Area Under Curve
    '''

    def __init__(self, task_type='binary', average='binary'):
        super(AUC, self).__init__()
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
        super(F1Score).__init__()
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
                self.value()
            else:
                thresh, f1 = self.thresh_search(y_prob=y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")
        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def value(self):
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'


class ClassificationReport(Metric):
    '''
    classification report
    '''

    def __init__(self, target_names=None):
        super(ClassificationReport).__init__()
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
        super(MultiLabelReport).__init__()
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
