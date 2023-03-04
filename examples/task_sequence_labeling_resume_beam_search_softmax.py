import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertTokenizer,
    BertPreTrainedModel,
    BertModel
)
from torch.nn import CrossEntropyLoss
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils.logger import Logger
from torchblocks.utils.options import Argparser
from torchblocks.utils.device import build_device
from torchblocks.utils import seed_everything
from torchblocks.tasks import ner_beam_search_decode
from torchblocks.utils import concat_tensors_with_padding, tensor_to_numpy
from torchblocks.tasks import get_spans_from_bio_tags
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore


class BertForTokenClassification(BertPreTrainedModel, Application):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def compute_loss(self, logits, labels, attention_mask):
        loss_fct = CrossEntropyLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
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
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, attention_mask)
        return {"loss": loss, "logits": logits, "attention_mask": attention_mask}


class SequenceLabelingTrainer(TrainBaseBuilder):
    def process_batch_outputs(self, tensor_dict):
        # beam_search搜索
        preds, pred_probs = ner_beam_search_decode(
            concat_tensors_with_padding(tensor_dict['logits'], padding_shape=(0, 0, 0, 1),
                                        padding_value=0).float().log_softmax(dim=-1),
            self.opts.id2label,
            self.opts.decode_beam_size,
        )
        labels = concat_tensors_with_padding(tensor_dict['labels'], padding_shape=(0, 1), padding_value=0)
        attention_masks = concat_tensors_with_padding(tensor_dict['attention_mask'], padding_shape=(0, 1),
                                                      padding_value=0)
        preds, pred_probs = tensor_to_numpy(preds), tensor_to_numpy(pred_probs)
        labels = tensor_to_numpy(labels)
        input_lens = tensor_to_numpy(attention_masks.sum(1))
        # Collect the NER entities for predictions and labels to calculate the F1 score.
        pred_entities_list, label_entities_list = [], []
        for pred, input_len, label in zip(preds, input_lens, labels):
            # Extract the NER entities from BIO-naming tags. Note that the
            pred_entities = get_spans_from_bio_tags([self.opts.id2label[x] for x in pred[:input_len]])
            pred_entities_list.append(pred_entities)
            # Of course, we will extract the entities for labels.
            label_entities = get_spans_from_bio_tags([self.opts.id2label[x] for x in label[:input_len]])
            label_entities_list.append(label_entities)
        return {"preds": pred_entities_list, "target": label_entities_list}


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

    def build_examples(self, data, data_type):
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
        tokens = example['tokens']
        labels = example['labels']
        encoder_txt = self.tokenizer(tokens,
                                     truncation=True,
                                     padding="max_length",
                                     return_tensors='pt',
                                     return_overflowing_tokens=True,
                                     is_split_into_words=True,
                                     max_length=self.max_sequence_length)
        encoder_txt = {k: v.squeeze(0) for k, v in encoder_txt.items()}
        input_ids = encoder_txt["input_ids"]
        token_type_ids = encoder_txt["token_type_ids"]
        attention_mask = encoder_txt["attention_mask"]
        overflowing_tokens = encoder_txt["overflowing_tokens"]
        label_ids = None
        if labels is not None:
            truncate_len = len(tokens) - overflowing_tokens.size(-1)
            labels = ['O'] + labels[: truncate_len] + ['O']
            labels = labels + ['O'] * (self.max_sequence_length - truncate_len - 2)
            label_ids = [self.label2id[label] for label in labels]
            label_ids = torch.tensor(label_ids)
        inputs = {
            "input_ids": input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids
        }
        return inputs


def load_data(opts, file_name, data_type, tokenizer, max_sequence_length):
    process_piplines = [
        ProcessExample2Feature(
            ResumeDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return ResumeDataset(opts, file_name, data_type, process_piplines)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
}


def main():
    parser = Argparser().build_parser()
    group = parser.add_argument_group(title="beam search", description="bs")
    group.add_argument("--decode_beam_size", type=int, default=2)
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
    opts.num_labels = len(ResumeDataset.label2id())
    opts.label2id = ResumeDataset.label2id()
    opts.id2label = ResumeDataset.id2label()

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
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
