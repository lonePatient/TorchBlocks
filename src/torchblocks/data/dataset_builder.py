import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from ..utils.io_utils import check_file, is_file
from ..utils.common_utils import convert_to_list

logger = logging.getLogger()


class DatasetBaseBuilder(Dataset):
    # 在collect_fn不做任务操作，直接以list方式合并
    keys_to_ignore_on_collate_batch = []
    # 动态batch处理过程中需要进行按照batch最长长度进行截取的keys
    keys_to_dynamical_truncate_on_padding_batch = ['input_ids', 'attention_mask', 'token_type_ids']

    def __init__(self, opts, file_name, data_type, process_piplines, cached_feature_file=None):
        super().__init__()
        self.data_type = data_type
        self.data_dir = opts.data_dir
        self.max_examples = opts.max_examples
        self.use_data_cache = opts.use_data_cache
        self.dynamical_padding = opts.dynamical_padding
        self.cached_feature_file = cached_feature_file
        self.process_piplines = convert_to_list(process_piplines)
        if not is_file(file_name):
            file_name = os.path.join(self.data_dir, file_name)
            check_file(file_name)
        self.examples = self.build_examples(self.read_data(file_name), data_type)
        if self.max_examples is not None:
            logger.info(f'[Debug]: use {self.max_examples} examples. ')
            self.examples = self.examples[: self.max_examples]
        if self.use_data_cache:
            self.build_feature_cache(opts)

    def build_feature_cache(self, opts):
        if not is_file(self.cached_feature_file):
            if self.cached_feature_file is None:
                prefix = f'{opts.task_name}_{opts.model_type}_{self.data_type}_{opts.experiment_name}'
                self.cached_feature_file = prefix + '_feature.cache'
            self.cached_feature_file = os.path.join(self.data_dir, self.cached_feature_file)
        if not opts.overwrite_data_cache:
            logger.info(f"Loading features from cached file: {self.cached_feature_file}")
            self.features = torch.load(self.cached_feature_file)
        else:
            logger.info(f"Creating features from dataset file: {self.data_dir}")
            self.features = [
                self.process_example(example) for example in
                tqdm(self.examples, total=len(self.examples), desc="Converting examples to features......")]
            logger.info(f"Saving features to cached file: {self.cached_feature_file}")
            torch.save(self.features, self.cached_feature_file)

    def process_example(self, example):
        for proc in self.process_piplines:
            if proc is None: continue
            example = proc(example)
        return example

    def process_collator(self, batch, max_input_length):
        # 动态padding
        if self.dynamical_padding:
            for k in self.keys_to_dynamical_truncate_on_padding_batch:
                if k not in batch: continue
                if batch[k].dim() >= 2: batch[k] = batch[k][:, : max_input_length]
        return batch

    def build_data_collator(self, features):
        batch = {}
        first = features[0]
        max_input_length = first['input_ids'].size(0)
        if self.dynamical_padding:
            max_input_length = max([torch.sum(f["attention_mask"]) for f in features])
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None:
                if k in self.keys_to_ignore_on_collate_batch:
                    bv = [f[k] for f in features]
                else:
                    bv = torch.stack([f[k] for f in features]) if isinstance(v, torch.Tensor) else torch.tensor(
                        [f[k] for f in features])
                batch[k] = bv
        batch = self.process_collator(batch, max_input_length)
        return batch

    def __getitem__(self, index):
        if self.use_data_cache:
            feature = self.features[index]
        else:
            feature = self.process_example(self.examples[index])
        return feature

    def __len__(self):
        return len(self.examples)

    @classmethod
    def label2id(cls):
        labels = cls.get_labels()
        if not isinstance(labels, dict):
            return {label: i for i, label in enumerate(labels)}
        return labels

    @classmethod
    def id2label(cls):
        labels = cls.get_labels()
        if isinstance(labels, dict):
            return {value: key for key, value in labels.items()}
        return {i: label for i, label in enumerate(labels)}

    @staticmethod
    def get_labels():
        raise NotImplementedError('Method [DatasetBaseBuilder.get_labels] should be implemented.')

    def read_data(self, input_file):
        raise NotImplementedError('Method [DatasetBaseBuilder.read_data] should be implemented.')

    def build_examples(self, data, data_type):
        raise NotImplementedError('Method [DatasetBaseBuilder.build_examples] should be implemented.')
