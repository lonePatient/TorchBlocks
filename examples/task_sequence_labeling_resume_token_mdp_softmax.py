import torch
from typing import *
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchblocks.data import DatasetBaseBuilder
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.utils.device import build_device
from torchblocks.utils import seed_everything
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer


class BertSoftmaxForSeqLabel(BertPreTrainedModel, Application):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def compute_loss(self, outputs, target, attention_mask):
        loss_fct = CrossEntropyLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, self.num_labels)[active_loss]
        active_labels = target.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)
        return loss

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs.get("labels", None)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        logits1 = self.classifier(self.dropout1(sequence_output))
        logits2 = self.classifier(self.dropout2(sequence_output))
        logits3 = self.classifier(self.dropout3(sequence_output))
        logits4 = self.classifier(self.dropout4(sequence_output))
        logits5 = self.classifier(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)
        loss, groundtruths, predictions = None, None, None
        if labels is not None:
            loss1 = self.compute_loss(logits1, labels, attention_mask)
            loss2 = self.compute_loss(logits2, labels, attention_mask)
            loss3 = self.compute_loss(logits3, labels, attention_mask)
            loss4 = self.compute_loss(logits4, labels, attention_mask)
            loss5 = self.compute_loss(logits5, labels, attention_mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            if not self.training:
                groundtruths = self.decode(labels, attention_mask, is_logits=False)
        if not self.training:  # 训练时无需解码
            predictions = self.decode(logits, attention_mask, is_logits=True)
        return {
            "loss": loss,
            "logits": logits,
            "predictions": predictions,
            "groundtruths": groundtruths
        }

    def decode(self, logits, mask, is_logits=False) -> List[List[List[str]]]:
        decode_ids = logits
        if is_logits:
            decode_ids = torch.argmax(logits, -1)  # (batch_size, seq_length)
        decode_labels = []
        for ids, mask in zip(decode_ids, mask):
            decode_label = [self.config.id2label[id.item()] for id, m in zip(ids, mask) if m > 0][1:-1]  # [CLS], [SEP]
            decode_labels.append(decode_label)
        return decode_labels


class ResumeDataset(DatasetBaseBuilder):
    keys_to_dynamical_truncate_on_padding_batch = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']

    @staticmethod
    def get_labels():
        labels = ['NAME', 'ORG', 'TITLE', 'RACE', 'EDU', 'CONT', 'LOC', 'PRO', ]
        Bio_labels = ["O"] + [f"B-{x}" for x in labels] + [f"I-{x}" for x in labels]
        return Bio_labels

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

    def build_examples(self, data, data_type, **kwargs):
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            tokens = line[0]
            labels = line[1] if data_type != 'test' else None
            examples.append(dict(guid=guid, tokens=tokens, labels=labels))
        return examples


class ProcessExample2Feature:

    def __init__(self, label2id, tokenizer, max_sequence_length):
        super().__init__()
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, example):
        text = example['tokens']
        labels = example['labels']
        encoding = self.tokenizer(text,
                                  padding="max_length",
                                  truncation="longest_first",
                                  max_length=self.max_sequence_length,
                                  return_overflowing_tokens=True,
                                  is_split_into_words=True,
                                  return_tensors='pt',
                                  )

        overflowing_tokens = encoding.pop("overflowing_tokens")
        num_truncated_tokens = encoding.pop("num_truncated_tokens")
        outputs = {k: v.squeeze(0) for k, v in encoding.items()}
        truncate_len = len(text) - overflowing_tokens.size(-1)
        padd_len = self.max_sequence_length - truncate_len - 2
        labels = ['O'] + labels[: truncate_len] + ['O']
        labels = labels + ['O'] * padd_len
        outputs["label_ids"] = torch.tensor([self.label2id[x] for x in labels])
        return outputs


def load_data(opts, file_name, data_type, tokenizer, max_sequence_length, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            ResumeDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return ResumeDataset(opts, file_name, data_type, process_piplines, **kwargs)


class SequenceLabelingTrainer(TrainBaseBuilder):

    def process_batch_outputs(self, batches):
        preds = []
        target = []
        for x,y in zip(batches['predictions'],batches['groundtruths']):
            preds.extend(x)
            target.extend(y)
        return {"preds": preds, "target": target}


MODEL_CLASSES = {
    "bert": (BertConfig, BertSoftmaxForSeqLabel, BertTokenizer),
}


def main():
    opts = Argparser().build_arguments()
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
    config = config_class.from_pretrained(opts.pretrained_model_path,
                                          num_labels=opts.num_labels,
                                          label2id=opts.label2id,
                                          id2label=opts.id2label)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)

    # trainer
    logger.info("initializing traniner")
    labels = {label.split('-')[1] for label in ResumeDataset.get_labels() if '-' in label}
    metrics = [SequenceLabelingScore(labels=labels, average='micro', schema='BIO')]
    trainer = SequenceLabelingTrainer(opts=opts, model=model, metrics=metrics, logger=logger)
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
