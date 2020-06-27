import string
import logging
from .base import DataProcessor
from .utils import InputFeatures

logger = logging.getLogger(__name__)

class TextClassifierProcessor(DataProcessor):
    '''
    encode_mode: 预处理方式.
                ``one``:表示只有一个inputs
                ``pair``：表示两个inputs，一般针对siamese类型网络
                ``triple``： 表示三个inputs，一般针对triple 类型网络
            (default: ``one``)
    '''

    def __init__(self,
                 tokenizer,
                 data_dir,
                 prefix='',
                 encode_mode='one',
                 add_special_tokens=True,  # [CLS]XXXX[SEP] or [CLS]XXX[SEP]YYYY[SEP
                 pad_to_max_length=True):

        super().__init__(data_dir=data_dir,
                         tokenizer=tokenizer,
                         encode_mode=encode_mode,
                         prefix=prefix)
        self.pad_to_max_length = pad_to_max_length
        self.add_special_tokens = add_special_tokens

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
                        label_ids[i] = lb
                    else:
                        raise ValueError("multi label type: expected one of (str,float,int)")
            else:
                label_ids = example.label_ids
            if example.label is not None:
                if isinstance(example.label,(float,int)):
                    label = int(example.label)
                elif isinstance(example.label,str):
                    label = label_map[example.label]
                else:
                    raise ValueError("label type: expected one of (str,float,int)")
            else:
                label = example.label
            if label is not None:
                inputs['label'] = label
            if label_ids is not None:
                inputs['label_ids'] = label_ids
            if ex_index < 5:
                self.print_examples(**inputs)
            features.append(InputFeatures(**inputs))
        return features
