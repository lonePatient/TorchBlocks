import os
import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Callable, Any
from torchblocks.utils.paths import check_file, is_file

logger = logging.getLogger(__name__)


class DatasetBase(torch.utils.data.Dataset):
    keys_to_truncate_on_dynamic_batch = ['input_ids', 'attention_mask', 'token_type_ids']

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines: List[Callable],
                 max_examples: int = None,
                 use_cache: bool = False,
                 collate_dynamic: bool = True,
                 cached_features_file: str = None,
                 overwrite_cache: bool = False,
                 **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        file_path = data_name
        if not is_file(data_name): file_path = os.path.join(data_dir, data_name)
        check_file(file_path)
        self.examples = self.create_examples(self.read_data(file_path), data_type)
        if max_examples is not None: self.examples = self.examples[: max_examples]
        self.process_piplines = process_piplines if isinstance(process_piplines, list) else [process_piplines]
        self.num_examples = len(self.examples)
        self.num_labels = len(self.get_labels())
        self.use_cache = use_cache
        self.collate_dynamic = collate_dynamic
        self.cached_features_file = cached_features_file
        self.overwrite_cache = overwrite_cache
        if self.use_cache:
            if cached_features_file is None:
                cached_features_file = f'{data_type}.cache'
            cached_features_file = os.path.join(self.data_dir, cached_features_file)
            self.create_features_cache(cached_features_file)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.use_cache:
            feature = self.features[index]
        else:
            feature = self.process_example(self.examples[index])
        return feature

    def __len__(self):
        return self.num_examples

    def create_features_cache(self, cached_features_file):
        if is_file(cached_features_file) and not self.overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(self.cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {self.data_dir}")
            self.features = [
                self.process_example(example) for example in
                tqdm(self.examples, total=self.num_examples, desc="Converting examples to features...")]
            logger.info(f"Saving features to cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    @classmethod
    def get_labels(self) -> List[str]:
        raise NotImplementedError('Method [get_labels] should be implemented.')

    @classmethod
    def label2id(cls):
        return {label: i for i, label in enumerate(cls.get_labels())}

    @classmethod
    def id2label(cls):
        return {i: label for i, label in enumerate(cls.get_labels())}

    def read_data(self, input_file: str) -> Any:
        raise NotImplementedError('Method [read_data] should be implemented.')

    def create_examples(self, data: Any, set_type: str, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError('Method [create_examples] should be implemented.')

    def process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        for proc in self.process_piplines:
            if proc is None: continue
            example = proc(example)
        return example

    def collate_fn(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        first = features[0]
        max_input_length = first['input_ids'].size(0)
        if self.collate_dynamic:
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
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                bv = torch.stack([f[k] for f in features]) if isinstance(v, torch.Tensor) else torch.tensor(
                    [f[k] for f in features])
                batch[k] = bv
        if self.collate_dynamic:
            for k in self.keys_to_truncate_on_dynamic_batch:
                if k not in batch: continue
                if batch[k].dim() >= 2: batch[k] = batch[k][:, : max_input_length]
        return batch
