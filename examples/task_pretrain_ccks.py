import json
import torch
import numpy as np
import torch.nn as nn
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils import seed_everything
from torchblocks.utils.options import Argparser
from torchblocks.utils.device import build_device
from torchblocks.utils.logger import Logger
from torchblocks.metrics.classification.accuracy import Accuracy
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

'''
针对CCKS2021任务预训练
'''


class BertForMaskedLM(BertPreTrainedModel, Application):
    # _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def compute_loss(self, outputs, labels):
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(outputs.view(-1, self.config.vocab_size), labels.view(-1))
        return loss

    def forward(self, inputs):
        outputs = self.bert(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = None
        labels = inputs.get("labels", None)
        if labels is not None:
            masked_lm_loss = self.compute_loss(prediction_scores, labels)
        return {"loss": masked_lm_loss, "logits": prediction_scores}


class CCKSDataset(DatasetBaseBuilder):

    def __init__(self, opts, file_name, data_type, process_piplines, tokenizer, mlm_probability, max_seq_len, **kwargs):
        super().__init__(opts, file_name, data_type, process_piplines, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    @staticmethod
    def get_labels():
        '''
        在预训练中添加标签信息
        Returns:
        '''
        return {'不匹配': 0, '部分匹配': 1, '完全匹配': 2, -1: -1}

    def read_data(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    def build_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{line['text_id']}"
            if line['query'] == '':
                continue
            text_a = line['query']
            for c in line['candidate']:
                if c['text'] == '':
                    continue
                text_b = c['text']
                label = c['label'] if data_type != 'test' else -1
                examples.append(dict(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    # 默认都是0进行padding
    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _ngram_mask(self, input_ids, max_seq_len):
        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))
        if len(input_ids) <= 32:
            max_ngram = 2
        else:
            max_ngram = 3
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        np.random.shuffle(ngram_indexes)
        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def ngram_mask(self, input_ids_list, max_seq_len):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs, mask_labels):

        labels = inputs.clone()
        probability_matrix = mask_labels
        bs = inputs.shape[0]
        # word struct prediction
        for i in range(bs):
            tmp = []
            tmp_pro = []
            tmp_pro.extend([1] * 3)
            now_input = inputs[i]
            now_probability_matrix = probability_matrix[i]
            now_probability_matrix = now_probability_matrix.cpu().numpy().tolist()
            now_input = now_input.cpu().numpy().tolist()
            for j in range(len(now_input)):
                if now_input[j] == self.tokenizer.sep_token_id:
                    sep_index = j
            # we don't choose cls_ids, sep_ids, pad_ids
            choose_range = now_input[1:sep_index - 2]
            if len(choose_range) == 0:
                choose_range = now_input[1:5]
            rd_token = np.random.choice(choose_range)
            token_idx = now_input.index(rd_token)
            tmp.extend(now_input[token_idx:token_idx + 3])
            np.random.shuffle(tmp)
            now_input[token_idx:token_idx + 3] = tmp
            now_probability_matrix[token_idx:token_idx + 3] = tmp_pro
            now_input = torch.tensor(now_input)
            now_probability_matrix = torch.tensor(now_probability_matrix)
            inputs[i] = now_input
            probability_matrix[i] = now_probability_matrix
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def build_data_collator(self, features):
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*features))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)
        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }
        return data_dict


class ProcessEncodeText:
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, label2id, tokenizer, max_seq_len):
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, example):
        label = self.label2id.get(example["label"])
        encoding = self.tokenizer.encode_plus(text=example["text_a"], text_pair=example['text_b'],
                                              add_special_tokens=True,
                                              max_length=self.max_seq_len,
                                              truncation=True,
                                              truncation_strategy='longest_first')
        if label != -1:
            # 添加标签信息，即：[CLS]TEXT[SEP]LABEL[SEP]
            encoding['input_ids'] = encoding['input_ids'] + [label + 1] + [102]
            encoding['attention_mask'] = encoding['attention_mask'] + [1] + [1]
            encoding['token_type_ids'] = encoding['token_type_ids'] + [0] + [0]
        return encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask']


class PretrainTrainer(TrainBaseBuilder):
    pass


def load_data(opts, file_name, data_type, tokenizer, mlm_probability, max_seq_len):
    process_piplines = [ProcessEncodeText(CCKSDataset.label2id(), tokenizer, max_seq_len)]
    return CCKSDataset(opts, file_name, data_type, process_piplines, tokenizer, mlm_probability, max_seq_len)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer)
}


def main():
    parser = Argparser().build_parser()
    group = parser.add_argument_group(title="pretrain", description="")
    group.add_argument("--mlm_probability", type=float, default=0.4)
    opts = parser.build_args_from_parser(parser)
    logger = Logger(opts=opts)
    # device
    logger.info("initializing device")
    opts.device, opts.device_num = build_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    train_dataset = load_data(opts,
                              opts.train_input_file,
                              "train",
                              tokenizer,
                              opts.mlm_probability,
                              opts.train_max_seq_length)
    dev_dataset = load_data(opts,
                            opts.eval_input_file,
                            "dev",
                            tokenizer,
                            opts.mlm_probability,
                            opts.eval_max_seq_length)
    test_dataset = load_data(opts,
                             opts.test_input_file,
                             "test",
                             tokenizer,
                             opts.mlm_probability,
                             opts.eval_max_seq_length)
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path)
    config.update(
        {
            "layer_norm_eps": 1e-7
        }
    )
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = PretrainTrainer(opts=opts,
                              model=model,
                              metrics=Accuracy(task="multiclass", num_classes=config.vocab_size),
                              logger=logger
                              )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset,
                      dev_data=None,
                      state_to_save={'vocab': tokenizer},
                      train_with_add_datasets=[dev_dataset, test_dataset])


if __name__ == "__main__":
    main()
