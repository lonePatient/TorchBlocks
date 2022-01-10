import torch
import logging
from torchblocks.metrics.base import Metric
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrcoef as _MatthewsCorrcoef

logger = logging.getLogger(__name__)


class MattewsCorrcoef(_MatthewsCorrcoef, Metric):
    '''
    Matthews Correlation Coefficient
    '''

    def __init__(self, num_classes, threshold=0.5):
        super(MattewsCorrcoef, self).__init__(num_classes=num_classes, threshold=threshold)

    def reset(self):
        default = torch.zeros(self.num_classes, self.num_classes)
        self.add_state("confmat", default=default, dist_reduce_fx="sum")

    def value(self):
        score = self.compute()
        return score.item()

    def name(self):
        return 'mcc'
