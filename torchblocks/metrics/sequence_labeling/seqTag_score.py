import pandas as pd
from torchblocks.metrics.base import Metric
from torchblocks.metrics.sequence_labeling.precision_recall_fscore import precision_recall_fscore_support


class SequenceLabelingScore(Metric):

    def __init__(self, labels, schema=None, average="micro"):
        self.labels = labels
        self.schema = schema
        self.average = average
        self.reset()

    def update(self, preds, target):
        self.preds.extend(preds)
        self.target.extend(target)

    def value(self):
        columns = ["label", "precision", "recall", "f1", "support"]
        values = []
        for label in [self.average] + sorted(self.labels):
            p, r, f, s = precision_recall_fscore_support(
                self.target, self.preds, average=self.average, schema=self.schema,
                labels=None if label == self.average else [label])
            values.append([label, p, r, f, s])
        df = pd.DataFrame(values, columns=columns)
        f1 = df[df['label'] == self.average]['f1'].item()
        return {
            "df": df, f"f1_{self.average}": f1,  # for monitor
        }

    def name(self):
        return "seqTag"

    def reset(self):
        self.preds = []
        self.target = []