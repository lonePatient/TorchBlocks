import os
import torch
import string
import logging
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)
LOWERCASE_STRS = list(string.ascii_lowercase)  # 获取26个小写字


class DataProcessor:
    """Base class for processor converters
       data_dir: 数据目录
       tokenizer: tokenizer
       encode_mode: 预处理方式.
                ``one``: 表示只有一个inputs
                ``pair``: 表示两个inputs，一般针对siamese类型网络
                ``triple``: 表示三个inputs，一般针对triple类型网络
                (default: ``one``)
        add_special_tokens: 是否增加[CLS]XXX[SEP], default: True
        pad_to_max_length: 是否padding到最大长度, default: True
        truncate_label: 是否label进行阶段，主要在collect_fn函数中，一般针对sequence labeling任务中，default: False
    """

    def __init__(self, data_dir, tokenizer,
                 encode_mode='one',
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 truncate_label=False,
                 truncation_strategy="longest_first",
                 prefix='',**kwargs):
        self.prefix = prefix
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.encode_mode = encode_mode
        self.truncate_label = truncate_label
        self.add_special_tokens = add_special_tokens
        self.pad_to_max_length = pad_to_max_length
        self.truncation_strategy = truncation_strategy

        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.encode_mode not in ['one', 'pair', 'triple']:
            raise ValueError(" encode_mode type: expected one of (one,pair,triple)")

    def get_train_examples(self, data_path):
        """Gets a collection of `InputExample`s for the train set."""
        return self.create_examples(self.read_data(data_path), 'train')

    def get_dev_examples(self, data_path):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.create_examples(self.read_data(data_path), 'dev')

    def get_test_examples(self, data_path):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.create_examples(self.read_data(data_path), 'test')

    def encode_plus(self, text_a, text_b, max_length):
        '''
        if add_special_tokens is True:
            If the data is sentence Pair, -> [CLS]senA[SEP]senB[SEP]
            If data is single sentence -> [CLS]senA[SEP]
        if add_special_tokens is False:
            If the data is sentence Pair, -> senA+senB
            If data is single sentence -> senA
        '''
        inputs = self.tokenizer.encode_plus(text=text_a,
                                            text_pair=text_b,
                                            max_length=max_length,
                                            truncation_strategy=self.truncation_strategy,
                                            add_special_tokens=self.add_special_tokens,
                                            pad_to_max_length=self.pad_to_max_length)
        return inputs

    def get_input_keys(self):
        '''
        inputs输入对应的keys，需要跟模型输入对应
        '''
        keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if self.encode_mode == 'one':
            return keys + ['labels']
        elif self.encode_mode == 'pair':
            return [f'{LOWERCASE_STRS[i]}_{item}' for i in range(2) for item in keys] + ['labels']
        elif self.encode_mode == 'triple':
            return [f'{LOWERCASE_STRS[i]}_{item}' for i in range(3) for item in keys] + ['labels']

    def encode(self, texts, max_seq_length):
        '''
        Args:
            texts: 列表形式,[text_a,text_b] or [[text_a1,text_b1],[text_a2,text_b2]],....
            max_seq_length: 最大长度
        Returns:
        '''
        inputs = {}
        if self.encode_mode == 'one':
            # texts:[text_a,text_b]
            assert len(texts) == 2, "texts length: expected to be 2"
            inputs = self.encode_plus(text_a=texts[0], text_b=texts[1], max_length=max_seq_length)
        elif self.encode_mode == 'pair':
            # texts:[[text_a1,text_b1],[text_a2,text_b2]]
            assert len(texts) == 2, "texts length: expected to be 2"
            for i in range(2):
                _inputs = self.encode_plus(text_a=texts[i][0], text_b=texts[i][1], max_length=max_seq_length)
                inputs.update(({f'{LOWERCASE_STRS[i]}_' + key: value for key, value in _inputs.items()}))
        elif self.encode_mode == 'triple':
            # texts:[[text_a1,text_b1],[text_a2,text_b2],[text_a3,text_b3]]
            assert len(texts) == 3, "texts length: expected to be 3"
            for i in range(3):
                _inputs = self.encode_plus(text_a=texts[i][0], text_b=texts[i][1], max_length=max_seq_length)
                inputs.update(({f'{LOWERCASE_STRS[i]}_' + key: value for key, value in _inputs.items()}))
        return inputs

    def collate_fn(self, batch):
        """
        batch数据动态长度变化（根据mask进行计算），batch形式必须满足：
        (input_ids, attention_mask, *,*,*, labels) tuples...
        """
        batch = list(map(torch.stack, zip(*batch)))
        max_seq_len = torch.max(torch.sum(batch[1], 1)).item()
        num_inputs = len(batch)
        # test时一般label为None，所以导致keys_num!=input_keys_num
        has_label = num_inputs == len(self.get_input_keys())
        for i in range(num_inputs):
            if batch[i].dim()==1:
                continue
            if i == num_inputs - 1 and has_label:# 最后一个是否为label
                if self.truncate_label:
                    batch[i] = batch[i][:, :max_seq_len]
            else:
                if batch[i].size(1) > max_seq_len:
                    batch[i] = batch[i][:, :max_seq_len]
        return batch

    def convert_to_tensors(self, features):
        # In this method we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if hasattr(first, "label") and first.label is not None:
            if type(first.label) is int:
                labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label for f in features], dtype=torch.float)
            batch = {"labels": labels}
        elif hasattr(first, "label_ids") and first.label_ids is not None:
            if type(first.label_ids[0]) is int:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
            batch = {"labels": labels}
        else:
            batch = {}
        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in vars(first).items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        return batch

    def load_from_cache(self, max_seq_length, data_name, mode):
        '''
        数据加载
        Args:
            max_seq_length: 最大长度
            data_name: 数据名称
            mode: 数据类型，可选['train', 'dev', 'test']
        Returns:
        '''
        # Load processor features from cache or dataset file
        cached_features_file = f'cached_{mode}_{self.prefix}_{max_seq_length}'
        cached_features_file = os.path.join(self.data_dir, cached_features_file)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", str(self.data_dir))
            label_list = self.get_labels()
            if mode == 'train':
                examples = self.get_train_examples(os.path.join(self.data_dir, data_name))
            elif mode == 'dev':
                examples = self.get_dev_examples(os.path.join(self.data_dir, data_name))
            else:
                examples = self.get_test_examples(os.path.join(self.data_dir, data_name))
            features = self.convert_to_features(examples=examples, label_list=label_list, max_seq_length=max_seq_length)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, max_seq_length, data_name, mode):
        '''
        将features转换为dataset
        '''
        features = self.load_from_cache(max_seq_length=max_seq_length, data_name=data_name, mode=mode)
        inputs = self.convert_to_tensors(features)
        inputs_keys = self.get_input_keys()
        dataset = TensorDataset(*[inputs[key] for key in inputs_keys if inputs.get(key, None) is not None])
        return dataset

    def print_examples(self, **kwargs):
        '''
        打印样本信息
        '''
        inputs = {}
        for key, value in kwargs.items():
            if key not in inputs and value is not None:
                inputs[key] = value
        logger.info("*** Example ***")
        for key, value in inputs.items():
            if isinstance(value, (float, int)):
                value = str(value)
            elif isinstance(value, list):
                value = " ".join([str(x) for x in value])
            elif isinstance(value, str):
                value = value
            else:
                raise ValueError("'value' value type: expected one of (float,int,str,list)")
            logger.info("%s: %s" % (str(key), value))

    def convert_to_features(self, examples, label_list, max_seq_length):
        '''
        转化为特征
        '''
        raise NotImplementedError()

    def create_examples(self, **kwargs):
        '''
        创建exmaples
        '''
        raise NotImplementedError()

    def read_data(self, input_file):
        '''
        读取数据
        '''
        raise NotImplementedError()

    def get_labels(self):
        """
        标签列表
        """
        raise NotImplementedError()



