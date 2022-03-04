import torch
import logging
from torchblocks.metrics.base import Metric
from torchmetrics.classification.accuracy import Accuracy as _Accuracy

logger = logging.getLogger(__name__)


class Accuracy(_Accuracy, Metric):
    '''
    Computes accuracy. Works with binary, multiclass, and multilabel data.
    Accepts logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.
    Args:
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
    '''

    def __init__(self, threshold=0.5,
                 num_classes=None,
                 average="micro",
                 ignore_index=None,
                 top_k=None,
                 multiclass=None):
        super(Accuracy, self).__init__(threshold=threshold,
                                       num_classes=num_classes,
                                       average=average,
                                       ignore_index=ignore_index,
                                       top_k=top_k,
                                       multiclass=multiclass)
        self.reset()

    def reset(self):
        self.tp = torch.zeros([], dtype=torch.long)
        self.fp = torch.zeros([], dtype=torch.long)
        self.tn = torch.zeros([], dtype=torch.long)
        self.fn = torch.zeros([], dtype=torch.long)

    def value(self):
        score = self.compute()
        return score.item()

    def name(self):
        return 'acc'
