import os
import torch
from torch.utils.data import TensorDataset


class DataProcessor:
    """Base class for processor converters """
    def __init__(self,
                 data_dir,
                 logger,
                 prefix='',
                 truncate_label=False,
                 **kwargs):
        self.logger = logger
        self.data_dir = data_dir
        self.prefix = prefix
        self.truncate_label = truncate_label

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_train_examples(self, data_path):
        """Gets a collection of `InputExample`s for the train set."""
        return self.create_examples(self.read_data(data_path), 'train')

    def get_dev_examples(self, data_path):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.create_examples(self.read_data(data_path), 'dev')

    def get_test_examples(self, data_path):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.create_examples(self.read_data(data_path), 'test')


    def get_labels(self):
        """Gets the list of labels for this processor set."""
        return None

    def get_batch_keys(self):
        '''
         batch keys should be a list of (input_ids, attention_mask, *,*,*, labels) tuples...
        '''
        raise NotImplementedError

    def convert_to_features(self, **kwargs):
        raise NotImplementedError()

    def create_examples(self, **kwargs):
        raise NotImplementedError()

    def read_data(self, input_file):
        raise NotImplementedError()

    def collate_fn(self, batch):

        """
        batch should be a list of (input_ids, attention_mask, *,*,*, labels) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        batch = list(map(torch.stack, zip(*batch)))
        max_seq_len = torch.max(torch.sum(batch[1], 1)).item()
        for i in range(len(batch) - 1):
            if batch[i].size()[1] > max_seq_len:
                batch[i] = batch[i][:, :max_seq_len]
        if self.truncate_label:
            batch[-1] = batch[-1][:, :max_seq_len]
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
        load feature cache
        '''
        # Load processor features from cache or dataset file
        cached_features_file = f'cached_{mode}_{self.prefix}_{max_seq_length}'
        cached_features_file = os.path.join(self.data_dir, cached_features_file)

        if os.path.exists(cached_features_file):
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self.logger.info("Creating features from dataset file at %s", str(self.data_dir))
            label_list = self.get_labels()
            if mode == 'train':
                examples = self.get_train_examples(os.path.join(self.data_dir, data_name))
            elif mode == 'dev':
                examples = self.get_dev_examples(os.path.join(self.data_dir, data_name))
            else:
                examples = self.get_test_examples(os.path.join(self.data_dir, data_name))
            features = self.convert_to_features(examples=examples, label_list=label_list, max_seq_length=max_seq_length)
            self.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, max_seq_length, data_name, mode):
        '''
        dataset
        '''
        features = self.load_from_cache(max_seq_length=max_seq_length, data_name=data_name, mode=mode)
        inputs = self.convert_to_tensors(features)
        inputs_keys = self.get_batch_keys()
        dataset = TensorDataset(*[inputs[key] for key in inputs_keys if inputs[key] is not None])
        return dataset

    def print_examples(self, **kwargs):
        '''
        打印一些样本信息
        '''
        inputs = {}
        for key, value in kwargs.items():
            if key not in inputs and value is not None:
                inputs[key] = value
        self.logger.info("*** Example ***")
        for key, value in inputs.items():
            if isinstance(value, (float, int)):
                value = str(value)
            elif isinstance(value, list):
                value = " ".join([str(x) for x in value])
            elif isinstance(value, str):
                value = value
            else:
                raise ValueError("'value' value type: expected one of (float,int,str,list)")
            self.logger.info("%s: %s" % (str(key), value))
