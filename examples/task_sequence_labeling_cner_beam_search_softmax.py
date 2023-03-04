import json
import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertTokenizerFast,
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
from torchblocks.tasks.sequence_tags import generate_bio_tags_from_spans
from torchblocks.tasks.sequence_tags import ner_beam_search_decode
from torchblocks.utils import concat_tensors_with_padding, tensor_to_numpy
from torchblocks.tasks.sequence_tags import get_spans_from_subword_bio_tags
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
    keys_to_ignore_on_gpu = ['offset_mapping', 'text']

    def process_batch_outputs(self, tensor_dict):
        texts = tensor_dict['text']  # 原始文本
        # beam_search搜索
        preds, pred_probs = ner_beam_search_decode(
            concat_tensors_with_padding(tensor_dict['logits'], padding_shape=(0, 0, 0, 1),
                                        padding_value=0).float().log_softmax(dim=-1),
            self.opts.id2label,
            self.opts.decode_beam_size,
        )
        labels = concat_tensors_with_padding(tensor_dict['labels'], padding_shape=(0, 1), padding_value=0)
        offset_mappings = concat_tensors_with_padding(tensor_dict['offset_mapping'], padding_shape=(0, 0,0,1),
                                                      padding_value=0)
        preds, pred_probs = tensor_to_numpy(preds), tensor_to_numpy(pred_probs)
        labels, offset_mappings = tensor_to_numpy(labels), tensor_to_numpy(offset_mappings)
        # Collect the NER entities for predictions and labels to calculate the F1 score.
        pred_entities_list, label_entities_list = [], []
        for text, pred, pred_prob, label, offset_mapping in zip(
                texts, preds, pred_probs, labels, offset_mappings
        ):
            valid_mask = offset_mapping[..., 1] > 0
            pred, pred_prob = pred[valid_mask], pred_prob[valid_mask]
            label, offset_mapping = label[valid_mask], offset_mapping[valid_mask]
            # Extract the NER entities from BIO-naming tags. Note that the
            # low-confidence or too-short entities will be dropped.
            pred_entities, pred_entity_probs = get_spans_from_subword_bio_tags(
                [self.opts.id2label[x] for x in pred], offset_mapping, pred_prob
            )
            pred_entities = [
                (entity, a, b)
                for (entity, a, b), prob in zip(pred_entities, pred_entity_probs)
            ]
            pred_entities_list.append(pred_entities)
            # Of course, we will extract the entities for labels.
            label_entities, _ = get_spans_from_subword_bio_tags(
                [self.opts.id2label[x] for x in label], offset_mapping
            )
            label_entities_list.append(label_entities)

        return {"preds": pred_entities_list, "target": label_entities_list}


class CnerDataset(DatasetBaseBuilder):
    keys_to_ignore_on_collate_batch = ['text']
    keys_to_dynamical_truncate_on_padding_batch = [
        'input_ids', 'attention_mask', 'token_type_ids', 'labels', 'offset_mapping'
    ]

    @staticmethod
    def get_labels():
        labels = ['CONT', 'EDU', 'LOC', 'NAME', 'ORG', 'PRO', 'RACE', 'TITLE']
        Bio_labels = ["O"] + [f"B-{x}" for x in labels] + [f"I-{x}" for x in labels]
        return Bio_labels

    def read_data(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    def build_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{line['id']}"
            text = line['text']
            entities = line['entities']
            examples.append(dict(guid=guid, text=text, entities=entities))
        return examples


class ProcessExample2Feature:

    def __init__(self, label2id, tokenizer, max_sequence_length):
        super().__init__()
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, example):
        text = example['text']
        entities = example['entities']
        ## 处理空格以及异常符号
        new_tokens = []
        for i, word in enumerate(text):
            tokenizer_word = self.tokenizer.tokenize(word)
            if len(tokenizer_word) == 0:
                new_tokens.append("^")
            else:
                new_tokens.append(word)
        new_text = "".join(new_tokens)
        encoding = self.tokenizer(new_text,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors='pt',
                                  max_length=self.max_sequence_length,
                                  return_offsets_mapping=True)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        outputs = dict(**encoding, text="".join(text))  # 保持原有的text，后续解码使用
        # 将[['PER', 0,1]]转化为['B-AGE', 'I-AGE', 'I-AGE', 'I-AGE',..........]
        bio_seq_tags = generate_bio_tags_from_spans(entities, encoding['offset_mapping'])
        outputs["label_ids"] = torch.tensor([self.label2id[x] for x in bio_seq_tags])
        return outputs


def load_data(opts, file_name, data_type, tokenizer, max_sequence_length, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            CnerDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return CnerDataset(opts, file_name, data_type, process_piplines, **kwargs)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizerFast),
}


def main():
    parser = Argparser().build_parser()
    group = parser.add_argument_group(title="add", description="")
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
    opts.num_labels = len(CnerDataset.label2id())
    opts.label2id = CnerDataset.label2id()
    opts.id2label = CnerDataset.id2label()
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
    labels = {label.split('-')[1] for label in CnerDataset.get_labels() if '-' in label}
    metrics = [SequenceLabelingScore(labels=labels, average='micro', schema='BIO')]
    trainer = SequenceLabelingTrainer(opts=opts, model=model, metrics=metrics, logger=logger)
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
