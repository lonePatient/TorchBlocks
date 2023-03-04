from ..base import Metric
from typing import List, Optional, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.auroc import AUROC as _AUROC


class AUC(Metric):
    '''
    Area Under Curve
    '''

    def __init__(self,
                 task: Literal["binary", "multiclass", "multilabel"],
                 thresholds: Optional[Union[int, List[float], Tensor]] = None,
                 num_classes: Optional[int] = None,
                 num_labels: Optional[int] = None,
                 average: Optional[Literal["macro", "weighted", "none"]] = "macro",
                 max_fpr: Optional[float] = None,
                 ignore_index: Optional[int] = None,
                 validate_args: bool = True,
                 ):
        self.task = task,
        self.thresholds = thresholds,
        self.num_classes = num_classes,
        self.num_labels = num_labels,
        self.average = average,
        self.max_fpr = max_fpr,
        self.ignore_index = ignore_index,
        self.validate_args = validate_args,
        self.reset()

    def reset(self):
        self.method = _AUROC(task=self.task,
                             thresholds=self.thresholds,
                             num_classes=self.num_classes,
                             num_labels=self.num_labels,
                             average=self.average,
                             max_fpr=self.max_fpr,
                             ignore_index=self.ignore_index,
                             validate_args=self.validate_args)

    def update(self, preds, target):
        self.method.update(preds, target)

    def value(self):
        score = self.method.compute()
        return score.item()

    def name(self):
        return 'auc'
