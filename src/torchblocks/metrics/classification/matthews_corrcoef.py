import torchmetrics
from ..base import Metric
from packaging import version
from typing import Optional
from typing_extensions import Literal

if version.parse(torchmetrics.__version__) >= version.parse("0.11.3"):
    from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef as _MatthewsCorrcoef
else:
    msg = ("The torchmetrics package version needs to be greater than 0.11.3, please update")
    raise ImportError(msg)


class MattewsCorrcoef(Metric):
    '''
    Matthews Correlation Coefficient
    '''

    def __init__(self, task: Literal["binary", "multiclass", "multilabel"] = None,
                 threshold: float = 0.5,
                 num_classes: Optional[int] = None,
                 num_labels: Optional[int] = None,
                 ignore_index: Optional[int] = None,
                 validate_args: bool = True
                 ):
        self.task = task
        self.threshold = threshold
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.reset()

    def reset(self):
        self.method = _MatthewsCorrcoef(task=self.task,
                                        threshold=self.threshold,
                                        num_classes=self.num_classes,
                                        num_labels=self.num_labels,
                                        ignore_index=self.ignore_index,
                                        validate_args=self.validate_args
                                        )

    def value(self):
        score = self.method.compute()
        return score.item()

    def update(self, preds, target):  # type: ignore
        self.method.update(preds, target)

    def name(self):
        return 'mcc'
