import torch
import logging
from torchblocks.metrics.base import Metric
from torchmetrics.classification.f_beta import F1 as _F1

logger = logging.getLogger(__name__)


class F1Score(_F1, Metric):
    '''
    F1 Score
    '''

    def __init__(self, num_classes=None,
                 threshold=0.5,
                 average="micro",
                 ignore_index=None,
                 top_k=None,
                 multiclass=None):
        super(F1Score, self).__init__(top_k=top_k,
                                      average=average,
                                      num_classes=num_classes,
                                      threshold=threshold,
                                      ignore_index=ignore_index,
                                      multiclass=multiclass)

        self.reset()

    def reset(self):
        default = lambda: []
        reduce_fn = None
        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default(), dist_reduce_fx=reduce_fn)

    def value(self):
        score = self.compute()
        return score.item()

    def name(self):
        return 'f1'
