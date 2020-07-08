import logging
from .base import DataProcessor
from .utils import InputFeatures

logger = logging.getLogger(__name__)


class TextClassifierProcessor(DataProcessor):
    def __init__(self, data_dir, tokenizer,
                 prefix='',
                 encode_mode='one',
                 truncate_label=False,
                 add_special_tokens=True,
                 pad_to_max_length=True):

        super().__init__(data_dir=data_dir,
                         prefix=prefix,
                         tokenizer=tokenizer,
                         encode_mode=encode_mode,
                         truncate_label=truncate_label,
                         add_special_tokens=add_special_tokens,
                         pad_to_max_length=pad_to_max_length)

    def convert_to_features(self, examples, label_list, max_seq_length):
        label_map = {label: i for i, label in enumerate(label_list)} if label_list is not None else {}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            if not isinstance(texts, list):
                raise ValueError(" texts type: expected one of (list,)")
            inputs = self.encode(texts, max_seq_length)
            inputs['guid'] = example.guid
            if example.label_ids is not None:
                label_ids = [0] * len(label_map)  # 多标签分类
                for i, lb in enumerate(example.label_ids):
                    if isinstance(lb, str):
                        label_ids[label_map[lb]] = 1
                    elif isinstance(lb, (float, int)):
                        label_ids[i] = int(lb)
                    else:
                        raise ValueError("multi label type: expected one of (str,float,int)")
            else:
                label_ids = example.label_ids
            if example.label is not None:
                if isinstance(example.label, (float, int)):
                    label = int(example.label)
                elif isinstance(example.label, str):
                    label = label_map[example.label]
                else:
                    raise ValueError("label type: expected one of (str,float,int)")
            else:
                label = example.label
            inputs['label'] = label
            inputs['label_ids'] = label_ids
            if ex_index < 5:
                self.print_examples(**inputs)
            features.append(InputFeatures(**inputs))
        return features
