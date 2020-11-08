import logging
from torchblocks.processor.base import DataProcessor
from torchblocks.processor.utils import InputFeatures

logger = logging.getLogger(__name__)


class SequenceLabelingProcessor(DataProcessor):
    '''
        special_token_label: [CLS]和[SEP]对应的标签, defalult: 'O'
        pad_label_id: padding对应的标签id, 默认使用'X',即default: 0
    '''

    def __init__(self, tokenizer, data_dir,
                 prefix='',
                 encode_mode='one',
                 truncate_label=True,
                 special_token_label='O',
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 pad_label_id=0):

        super().__init__(data_dir=data_dir,
                         prefix=prefix,
                         tokenizer=tokenizer,
                         encode_mode=encode_mode,
                         pad_to_max_length=pad_to_max_length,
                         add_special_tokens=add_special_tokens,
                         truncate_label=truncate_label)

        self.pad_label_id = pad_label_id
        self.special_token_label = special_token_label

    def convert_to_features(self, examples, label_list, max_seq_length, **kwargs):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            inputs = self.encode(texts=texts, max_seq_length=max_seq_length)
            label_ids = example.label_ids
            if label_ids is None or not isinstance(label_ids, list):
                raise ValueError("label_ids is not correct")
            special_toekns_num = 2 if self.add_special_tokens else 0
            if len(label_ids) > max_seq_length - special_toekns_num:  # [CLS] and [SEP]
                label_ids = label_ids[:(max_seq_length - special_toekns_num)]
            label_ids = [label_map[x] for x in label_ids]
            label_ids = [label_map[self.special_token_label]] + label_ids + [label_map[self.special_token_label]]
            label_ids += [self.pad_label_id] * (max_seq_length - len(label_ids))  # padding
            inputs['guid'] = example.guid
            inputs['label_ids'] = label_ids
            if ex_index < 5:
                self.print_examples(**inputs)
            features.append(InputFeatures(**inputs))
        return features


class SequenceLabelingSpanProcessor(DataProcessor):
    '''
    span sequence labeling
    '''
    def __init__(self, tokenizer, data_dir,
                 prefix='',
                 encode_mode='one',
                 truncate_label=True,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 pad_label_id=0):

        super().__init__(data_dir=data_dir,
                         prefix=prefix,
                         encode_mode=encode_mode,
                         tokenizer=tokenizer,
                         pad_to_max_length=pad_to_max_length,
                         add_special_tokens=add_special_tokens,
                         truncate_label=truncate_label)
        self.pad_label_id = pad_label_id

    def get_batch_keys(self):
        return ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']

    def convert_to_features(self, examples, label_list, max_seq_length):
        label2id = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            inputs = self.encode(texts=texts, max_seq_length=max_seq_length)
            start_positions = [self.pad_label_id] * max_seq_length
            end_positions = [self.pad_label_id] * max_seq_length
            for span in example.label_ids:
                label = span[0]
                if self.add_special_tokens:
                    start = span[1] + 1  # cls
                    end = span[2] + 1  # cls
                    special_num = 2
                else:
                    start = span[1]
                    end = span[2]
                    special_num = 0
                if start > max_seq_length - special_num:
                    continue
                start_positions[start] = label2id[label]
                if end > max_seq_length - special_num:
                    continue
                end_positions[end] = label2id[label]
            assert len(start_positions) == max_seq_length
            assert len(end_positions) == max_seq_length
            inputs['guid'] = example.guid
            inputs['start_positions'] = start_positions
            inputs['end_positions'] = end_positions
            if ex_index < 5:
                self.print_examples(**inputs)
            features.append(InputFeatures(**inputs))
        return features
