import torch
import torch.nn as nn
import torch.nn.functional as F
from torchblocks.losses.span_loss import SpanLoss
from transformers import BertPreTrainedModel, BertModel
from transformers import BertConfig, BertTokenizer
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.utils.device import build_device
from torchblocks.utils import seed_everything
from torchblocks.utils import tensor_to_list
from torchblocks.modules import PoolerStartLogits, PoolerEndLogits
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore
from torchblocks.tasks import get_spans_from_bio_tags


class BertSpanForSeqLabel(BertPreTrainedModel, Application):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, config.num_labels)
        self.end_fc = PoolerEndLogits(config.hidden_size + config.num_labels, config.num_labels)
        self.init_weights()

    def compute_loss(self, start_logits, end_logits, start_positions, end_positions, attention_mask):
        loss_fct = SpanLoss()
        loss = loss_fct(preds=(start_logits, end_logits),
                        target=(start_positions, end_positions),
                        masks=attention_mask)
        return loss

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        start_positions = inputs.get('start_positions', None)
        end_positions = inputs.get("end_positions", None)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if self.training:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            label_logits = torch.zeros([batch_size, seq_len, self.config.num_labels])
            label_logits = label_logits.to(input_ids.device)
            label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
        else:
            label_logits = F.softmax(start_logits, -1)
        end_logits = self.end_fc(sequence_output, label_logits)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.compute_loss(start_logits, end_logits, start_positions, end_positions, attention_mask)
        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}


class ResumeDataset(DatasetBaseBuilder):
    keys_to_ignore_on_collate_batch = ['entities']
    keys_to_dynamical_truncate_on_padding_batch = [
        'input_ids',
        'attention_mask',
        'token_type_ids',
        'start_positions',
        'end_positions'
    ]

    @staticmethod
    def get_labels():
        return ["O", "CONT", "ORG", "LOC", 'EDU', 'NAME', 'PRO', 'RACE', 'TITLE']

    def read_data(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words, labels = [], []
            for line in f:
                if line == "" or line == "\n":
                    if words:
                        lines.append([words, labels])
                        words, labels = [], []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        label = splits[-1].replace("\n", "")
                        labels.append(label)
                    else:
                        labels.append("O")
            if words:
                lines.append([words, labels])
        return lines

    def build_examples(self, data, data_type):
        examples = []
        for i, (words, labels) in enumerate(data):
            spans = get_spans_from_bio_tags(labels, id2label=None)
            new_spans = []
            for span in spans:
                tag, start, end = span
                new_spans.append([tag, start, end, "".join(words[start:(end + 1)])])  # 左闭右闭
            guid = f"{data_type}-{i}"
            entities = new_spans if data_type != 'test' else None
            examples.append(dict(guid=guid, tokens=words, entities=entities))
        return examples


class ProcessExample2Feature:

    def __init__(self, label2id, tokenizer, max_sequence_length):
        super().__init__()
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, example):
        tokens = example['tokens']
        entities = example['entities']
        inputs = self.tokenizer(
            tokens,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_sequence_length,
            return_overflowing_tokens=True,
            is_split_into_words=True,
            return_tensors='pt',
        )
        inputs.pop("overflowing_tokens")
        inputs.pop("num_truncated_tokens")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if entities is None:
            inputs['start_positions'] = None
            inputs['end_positions'] = None
            return inputs
        start_positions = [self.label2id["O"]] * self.max_sequence_length
        end_positions = [self.label2id["O"]] * self.max_sequence_length
        for label, start, end, *_ in entities:
            start += 1
            end += 1  # [CLS]
            label_id = self.label2id[label]
            if start < self.max_sequence_length and end < self.max_sequence_length:
                start_positions[start] = label_id
                end_positions[end] = label_id
        inputs['start_positions'] = torch.tensor(start_positions)
        inputs['end_positions'] = torch.tensor(end_positions)
        inputs['entities'] = entities
        return inputs


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            ResumeDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return ResumeDataset(data_name, data_dir, data_type, process_piplines, **kwargs)


class SequenceLabelingTrainer(TrainBaseBuilder):
    keys_to_ignore_on_gpu = ["entities"]  # batch数据中不转换为GPU的变量名
    keys_to_ignore_on_save_checkpoint = ['optimizer']  # checkpoint中不存储的模块，比如'optimizer'

    def process_batch_outputs(self, outputs):
        preds = []
        target = []
        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        attention_mask = outputs['attention_mask']
        entities = outputs['entities']  # batch列表数据
        for s_logit, e_logit, mask in zip(start_logits, end_logits, attention_mask):
            input_lens = tensor_to_list(torch.sum(mask, dim=-1))
            start_preds = tensor_to_list(torch.argmax(s_logit, -1))
            end_preds = tensor_to_list(torch.argmax(e_logit, -1))
            for s_pred, e_pred, le in zip(start_preds, end_preds, input_lens):
                s_pred = s_pred[:le][1:-1]
                e_pred = e_pred[:le][1:-1]
                p_ent = []
                for i, s_l in enumerate(s_pred):
                    if s_l == 0:
                        continue
                    for j, e_l in enumerate(e_pred[i:]):
                        if s_l == e_l:
                            p_ent.append((self.opts.id2label[s_l], i, i + j))
                            break
                preds.append(p_ent)
        for bd in entities:
            for b in bd:
                target.append([(x[0], x[1], x[2]) for x in b])
        return {"preds": preds, "target": target}


MODEL_CLASSES = {
    "bert": (BertConfig, BertSpanForSeqLabel, BertTokenizer),
}

def main():
    opts = Argparser.build_arguments()
    logger = Logger(opts=opts)

    # device
    logger.info("initializing device")
    opts.device, opts.device_num = build_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]

    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    train_dataset = load_data(opts, opts.train_input_file, "train", tokenizer, opts.train_max_seq_length)
    dev_dataset = load_data(opts, opts.eval_input_file, "dev", tokenizer, opts.eval_max_seq_length)

    opts.num_labels = len(ResumeDataset.label2id())
    opts.label2id = ResumeDataset.label2id()
    opts.id2label = ResumeDataset.id2label()

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    # FIXED: 默认`from_dict`中，只有config中有键才能设置值，这里强制设置
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    labels = [label for label in ResumeDataset.get_labels() if label != 'O']
    metrics = [SequenceLabelingScore(labels=labels, average='micro', schema='BIO')]
    trainer = SequenceLabelingTrainer(opts=opts,
                                      model=model,
                                      metrics=metrics,
                                      logger=logger,
                                      )
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
