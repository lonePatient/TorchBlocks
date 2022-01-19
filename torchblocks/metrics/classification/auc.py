import logging
from torchblocks.metrics.base import Metric
from torchmetrics.classification.auroc import AUROC as _AUROC

logger = logging.getLogger(__name__)


class AUC(_AUROC, Metric):
    '''
    Area Under Curve
    '''

    def __init__(self, num_classes=None,
                 pos_label=None,
                 average="macro"):
        super(AUC, self).__init__(num_classes=num_classes,
                                  pos_label=pos_label,
                                  average=average)
        self.reset()

    def reset(self):
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def value(self):
        score = self.compute()
        return score.item()

    def name(self):
        return 'auc'
