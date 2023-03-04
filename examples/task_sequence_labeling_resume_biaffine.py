import torch
import torch.nn as nn
import numpy as np
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.utils.device import build_device
from torchblocks.utils import seed_everything
from transformers import BertTokenizerFast, BertConfig
from transformers import BertPreTrainedModel, BertModel
from torchblocks.utils import tensor_to_numpy
from torchblocks.modules.biaffine import Biaffine
from torchblocks.tasks.sequence_tags import get_spans_from_bio_tags
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore


class BertBiaffineForSeqLabel(BertPreTrainedModel, Application):

    def __init__(self, config):
        super(BertBiaffineForSeqLabel, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.biaffine_bias = config.biaffine_bias
        self.biaffine_ffnn_size = config.biaffine_ffnn_size
        self.bert = BertModel(config)
        self.active = nn.ELU()
        self.start_mlp = nn.Linear(self.hidden_size, self.biaffine_ffnn_size)
        self.end_mlp = nn.Linear(self.hidden_size, self.biaffine_ffnn_size)
        self.biaffine = Biaffine(self.biaffine_ffnn_size, self.num_labels,
                                 bias=(self.biaffine_bias, self.biaffine_bias))
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs.get("labels", None)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_feat = self.active(self.start_mlp(sequence_output))
        end_feat = self.active(self.end_mlp(sequence_output))
        logits = self.biaffine(start_feat, end_feat)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, attention_mask)
        return {"loss": loss, "logits": logits, 'attention_mask': attention_mask}

    def compute_loss(self, logits, labels, mask):
        label_mask = torch.triu(mask.unsqueeze(-1).expand_as(labels).clone())
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.num_labels),
            labels.masked_fill(~label_mask.bool(), -100).reshape(-1),
        )
        return loss


class ResumeDataset(DatasetBaseBuilder):
    keys_to_dynamical_truncate_on_padding_batch = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']

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
                new_spans.append([tag, start, end + 1, "".join(words[start:(end + 1)])])
            guid = f"{data_type}-{i}"
            entities = new_spans if data_type != 'test' else None
            examples.append(dict(guid=guid, tokens=words, entities=entities))
        return examples

    def process_collator(self, batch, max_input_length):
        # 动态padding
        if self.dynamical_padding:
            for k in self.keys_to_dynamical_truncate_on_padding_batch:
                if k in batch:
                    if k in ['labels']:
                        batch[k] = batch[k][:, :max_input_length, :max_input_length]
                    elif batch[k].ndim == 2:
                        batch[k] = batch[k][:, : max_input_length]
        return batch


class ProcessExample2Feature:
    def __init__(self, label2id, tokenizer, max_sequence_length):
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, example):
        tokens = example['tokens']
        entities = example['entities']
        encoder_txt = self.tokenizer(
            tokens,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
            max_length=self.max_sequence_length,
            return_tensors='pt',
        )
        encoder_txt = {k: v.squeeze(0) for k, v in encoder_txt.items()}
        labels = torch.zeros((self.max_sequence_length, self.max_sequence_length), dtype=torch.long)
        for e_type, start_idx, end_idx, *_ in entities:
            if start_idx > self.max_sequence_length - 1 or end_idx > self.max_sequence_length - 1:
                continue
            labels[start_idx + 1, end_idx + 1] = self.label2id[e_type]
        inputs = {
            "input_ids": encoder_txt["input_ids"],
            'token_type_ids': encoder_txt["token_type_ids"],
            'attention_mask': encoder_txt["attention_mask"],
            'labels': labels
        }
        return inputs


class SequenceLabelingTrainer(TrainBaseBuilder):
    keys_to_ignore_on_save_checkpoint = ['optimizer']  # checkpoint中不存储的模块，比如'optimizer'

    def process_batch_outputs(self, outputs):
        """
        :param span_scores: (b, t, t, c)
        :param mask: (b, t)
        :return:
        """
        preds = []
        targets = []
        for logits, labels, attention_mask in zip(outputs['logits'], outputs['labels'], outputs['attention_mask']):
            predict_labels = torch.argmax(logits, -1)
            input_lens = torch.sum(attention_mask, dim=-1)
            mask = torch.tril(torch.ones_like(predict_labels), diagonal=-1)
            predict_labels = predict_labels - mask * 1e12
            y_pred = tensor_to_numpy(predict_labels)
            y_true = tensor_to_numpy(labels)
            pred = []
            target = []
            for b, start, end in zip(*np.where(y_pred > 0)):
                if start > input_lens[b] or end > input_lens[b]:
                    continue
                pred.append((self.opts.id2label[int(y_pred[b, start, end])], (b, start), (b, end)))
            for b, start, end in zip(*np.where(y_true > 0)):
                target.append((self.opts.id2label[int(y_true[b, start, end])], (b, start), (b, end)))
            preds.append(pred)
            targets.append(target)
        return {"preds": preds, "target": targets}


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            ResumeDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return ResumeDataset(data_name, data_dir, data_type, process_piplines, **kwargs)


MODEL_CLASSES = {
    "bert": (BertConfig, BertBiaffineForSeqLabel, BertTokenizerFast),
}


def main():
    parser = Argparser.build_parser()
    group = parser.add_argument_group(title="Biaffine", description="Biaffine")
    group.add_argument('--biaffine_ffnn_size', default=512, type=int)
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
    train_dataset = load_data(opts, opts.train_input_file, "train", tokenizer, opts.train_max_seq_length)
    dev_dataset = load_data(opts, opts.eval_input_file, "dev", tokenizer, opts.eval_max_seq_length)

    opts.num_labels = len(ResumeDataset.get_labels())
    opts.label2id = ResumeDataset.label2id()
    opts.id2label = ResumeDataset.id2label()
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    config.biaffine_ffnn_size = opts.biaffine_ffnn_size
    config.biaffine_bias = True
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    metrics = [SequenceLabelingScore(ResumeDataset.get_labels(), average='micro')]
    trainer = SequenceLabelingTrainer(opts=opts,
                                      model=model,
                                      metrics=metrics,
                                      logger=logger
                                      )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
