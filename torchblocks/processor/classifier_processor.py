from .base import DataProcessor
from .utils import InputFeatures
import string

LOWERCASE_STRS = list(string.ascii_lowercase)  # 获取26个小写字


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
                 logger,
                 prefix='',
                 encode_mode='one',
                 add_special_tokens=True,  # [CLS]XXXX[SEP] or [CLS]XXX[SEP]YYYY[SEP
                 pad_to_max_length=True):

        super().__init__(data_dir=data_dir,
                         logger=logger,
                         prefix=prefix)
        self.tokenizer = tokenizer
        self.pad_to_max_length = pad_to_max_length
        self.add_special_tokens = add_special_tokens
        self.encode_mode = encode_mode

        if self.encode_mode not in ['one', 'pair', 'triple']:
            raise ValueError(" encode_mode type: expected one of (one,pair,triple)")

    def get_batch_keys(self):
        if self.encode_mode == 'one':
            return ['input_ids', 'attention_mask', 'token_type_ids',
                    'labels']
        elif self.encode_mode == 'pair':
            return ['a_input_ids', 'a_attention_mask', 'a_token_type_ids',
                    'b_input_ids', 'b_attention_mask', 'b_token_type_ids',
                    'labels']
        else:
            return ['a_input_ids', 'a_attention_mask', 'a_token_type_ids',
                    'b_input_ids', 'b_attention_mask', 'b_token_type_ids',
                    'c_input_ids', 'c_attention_mask', 'c_token_type_ids',
                    'labels']

    def enocde(self, texts, max_seq_length):
        inputs = {}
        if self.encode_mode == 'one':
            # texts:[text_a,text_b]
            assert len(texts) == 2, "texts length: expected to be 2"
            inputs = self.tokenizer.encode_plus(text=texts[0],
                                                text_pair=texts[1],
                                                max_length=max_seq_length,
                                                add_special_tokens=self.add_special_tokens,
                                                pad_to_max_length=self.pad_to_max_length)
        elif self.encode_mode == 'pair':
            # texts:[[text_a,text_b],[text_a,text_b]]
            assert len(texts) == 2, "texts length: expected to be 2"
            for i in range(2):
                _inputs = self.tokenizer.encode_plus(text=texts[i][0],
                                                     text_pair=texts[i][1],
                                                     max_length=max_seq_length,
                                                     add_special_tokens=self.add_special_tokens,
                                                     pad_to_max_length=self.pad_to_max_length)
                inputs.update(({f'{LOWERCASE_STRS[i]}_' + key: value for key, value in _inputs.items()}))
        elif self.encode_mode == 'triple':
            # texts:[[text_a,text_b],[text_a,text_b],[text_a,text_b]]
            assert len(texts) == 3, "texts length: expected to be 3"
            for i in range(3):
                _inputs = self.tokenizer.encode_plus(text=texts[i][0],
                                                     text_pair=texts[i][1],
                                                     max_length=max_seq_length,
                                                     add_special_tokens=self.add_special_tokens,
                                                     pad_to_max_length=self.pad_to_max_length)
                inputs.update(({f'{LOWERCASE_STRS[i]}_' + key: value for key, value in _inputs.items()}))
        return inputs

    def convert_to_features(self, examples, label_list, max_seq_length):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                self.logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            if not isinstance(texts, list):
                raise ValueError(" texts type: expected one of (list)")
            inputs = self.enocde(texts, max_seq_length)
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
                label = label_map[example.label]
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
