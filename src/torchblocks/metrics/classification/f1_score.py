from ..base import Metric
from typing import Optional
from typing_extensions import Literal
from torchmetrics.classification.f_beta import F1Score as _F1Score


class F1Score(Metric):
    '''
    F1 Score
    '''

    def __init__(self, task: Literal["binary", "multiclass", "multilabel"],
                 threshold: float = 0.5,
                 num_classes: Optional[int] = None,
                 num_labels: Optional[int] = None,
                 average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
                 multidim_average: Optional[Literal["global", "samplewise"]] = "global",
                 top_k: Optional[int] = 1,
                 ignore_index: Optional[int] = None,
                 validate_args: bool = True,
                 ):
        self.task = task
        self.threshold = threshold
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.average = average
        self.multidim_average = multidim_average
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.reset()

    def reset(self):
        self.method = _F1Score(task=self.task,
                               threshold=self.threshold,
                               num_classes=self.num_classes,
                               num_labels=self.num_labels,
                               average=self.average,
                               multidim_average=self.multidim_average,
                               top_k=self.top_k,
                               ignore_index=self.ignore_index,
                               validate_args=self.validate_args, )

    def update(self, preds, target):
        self.method.update(preds, target)

    def value(self):
        score = self.method.compute()
        return score.item()

    def name(self):
        return 'f1'
