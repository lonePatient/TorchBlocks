from .base import DataProcessor
from .utils import InputFeatures


class SequenceLabelingProcessor(DataProcessor):
    '''
    sequence labeling
    '''
    def __init__(self,
                 tokenizer,
                 data_dir,
                 logger,
                 prefix='',
                 truncate_label=True,
                 special_token_label='O',
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 pad_label_id=0):

        super().__init__(data_dir=data_dir,
                         logger=logger,
                         prefix=prefix,
                         truncate_label=truncate_label)
        self.tokenizer = tokenizer
        self.pad_label_id = pad_label_id
        self.special_token_label = special_token_label
        self.add_special_tokens = add_special_tokens
        self.pad_to_max_length = pad_to_max_length

    def get_batch_keys(self):
        return ['input_ids', 'attention_mask', 'token_type_ids', 'labels']

    def convert_to_features(self, examples, label_list, max_seq_length,**kwargs):
            label_map = {label: i for i, label in enumerate(label_list)}
            features = []
            for (ex_index, example) in enumerate(examples):
                if ex_index % 10000 == 0:
                    self.logger.info("Writing example %d/%d" % (ex_index, len(examples)))
                inputs = self.tokenizer.encode_plus(text=example.text_a,
                                                    text_pair=example.text_b,
                                                    max_length=max_seq_length,
                                                    add_special_tokens=self.add_special_tokens,
                                                    pad_to_max_length=self.pad_to_max_length
                                                    )
                # label
                label_ids = example.label_ids
                special_toekns_num = 2 if self.add_special_tokens else 0
                if len(label_ids) > max_seq_length - special_toekns_num:  # [CLS] and [SEP]
                    label_ids = label_ids[:(max_seq_length - special_toekns_num)]
                label_ids = [label_map[x] for x in label_ids]
                label_ids = [label_map[self.special_token_label]] + label_ids + [label_map[self.special_token_label]]
                label_ids += [self.pad_label_id] * (max_seq_length - len(label_ids))

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
    def __init__(self,
                 tokenizer,
                 data_dir,
                 logger,
                 prefix='',
                 truncate_label=True,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 pad_label_id=0):

        super().__init__(data_dir=data_dir,
                         logger=logger,
                         prefix=prefix,
                         truncate_label=truncate_label)

        self.tokenizer = tokenizer
        self.pad_label_id = pad_label_id
        self.add_special_tokens = add_special_tokens
        self.pad_to_max_length = pad_to_max_length

    def get_batch_keys(self):
        return ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions','end_positions']

    def convert_to_features(self, examples, label_list, max_seq_length,**kwargs):
        label2id = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                self.logger.info("Writing example %d/%d" % (ex_index, len(examples)))

            inputs = self.tokenizer.encode_plus(text=example.text_a,
                                                text_pair=example.text_b,
                                                max_length=max_seq_length,
                                                add_special_tokens=self.add_special_tokens,
                                                pad_to_max_length=self.pad_to_max_length)

            start_positions = [self.pad_label_id] * max_seq_length
            end_positions = [self.pad_label_id] * max_seq_length
            for span in example.label_ids:
                label = span[0]
                start = span[1] + 1  # cls
                end = span[2] + 1  # cls
                if end >= max_seq_length:
                    continue
                start_positions[start] = label2id[label]
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

