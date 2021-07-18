from collections import Counter
from torchblocks.metrics.base import Metric
from torchblocks.metrics.utils_ner import *


class SequenceLabelingScore(Metric):

    def __init__(self, id2label, markup='bios', is_spans=False):
        self.id2label = id2label
        self.markup = markup
        self.is_spans = is_spans
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def update(self, input, target):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]
        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            # is_spans = False
            >>> target = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> input = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            # is_spans = True
            >>> target = [['MISC',2,6], ['MISC',3,6],['MISC',10,15]]
            >>> input = [['MISC',2,6], ['MISC',3,6],['MISC',10,15]]
        '''
        if self.is_spans:
            self.origins.extend(target)
            self.founds.extend(input)
            self.rights.extend([pre_entity for pre_entity in input if pre_entity in target])
        else:  # 针对crf编码之后的标签序列
            for label_path, pre_path in zip(target, input):
                label_entities = get_spans(label_path, self.id2label, self.markup)
                pre_entities = get_spans(pre_path, self.id2label, self.markup)
                self.origins.extend(label_entities)
                self.founds.extend(pre_entities)
                self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def value(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            type_ = self.id2label[type_] if isinstance(type_, (float, int)) else type_
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def name(self):
        pass
