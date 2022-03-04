from .base import Metric

from .sequence_labeling.scheme import get_scheme
from .sequence_labeling.precision_recall_fscore import precision_recall_fscore_support
from .sequence_labeling.seqTag_score import SequenceLabelingScore

from .classification.auc import AUC
from .classification.f1_score import F1Score
from .classification.accuracy import Accuracy
from .classification.matthews_corrcoef import MattewsCorrcoef