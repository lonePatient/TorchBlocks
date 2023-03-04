import torch
import numpy as np
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.utils.device import build_device
from torchblocks.utils import seed_everything
from torchblocks.utils import tensor_to_numpy
from transformers import (
    BertConfig,
    BertTokenizer,
    BertPreTrainedModel,
    BertModel
)
from torchblocks.utils import concat_tensors_with_padding
from torchblocks.metrics.base import Metric
from torchblocks.modules.global_pointer import GlobalPointer
from torchblocks.tasks import get_spans_from_bio_tags


class BertGlobalPointerForSeqLabel(BertPreTrainedModel, Application):
    def __init__(self, config):
        super(BertGlobalPointerForSeqLabel, self).__init__(config)
        self.num_labels = config.num_labels
        self.inner_dim = config.inner_dim
        self.use_rope = config.use_rope
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.global_pointer = GlobalPointer(self.num_labels, self.inner_dim, self.hidden_size, self.use_rope)
        self.dropout = torch.nn.Dropout(0.1)
        self.init_weights()

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs.get("labels", None)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # import pdb
        # pdb.set_trace()
        sequence_output = self.dropout(sequence_output)
        logits = self.global_pointer(sequence_output, mask=attention_mask)
        loss = None
        if labels is not None:
            loss = self.global_pointer.compute_loss(logits, labels)
            # loss = self.global_pointer.compute_loss(labels,logits)
        return {"loss": loss, "logits": logits}


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
                new_spans.append([tag, start, end, "".join(words[start:(end + 1)])])  # 左闭右闭
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
                        batch[k] = batch[k][:, :, :max_input_length, :max_input_length]
                    elif batch[k].dim() >= 2:
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
        input_ids = encoder_txt["input_ids"]
        token_type_ids = encoder_txt["token_type_ids"]
        attention_mask = encoder_txt["attention_mask"]
        labels = torch.zeros((len(self.label2id), self.max_sequence_length, self.max_sequence_length),
                             dtype=torch.int)
        for label, start, end, _ in entities:
            if start > self.max_sequence_length - 1 or end > self.max_sequence_length - 1:
                continue
            labels[self.label2id[label], start + 1, end + 1] = 1
        inputs = {
            "input_ids": input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs


class GobalPointerMetric(Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.preds = []
        self.target = []

    def update(self, preds, target):
        self.preds.extend(preds)
        self.target.extend(target)

    def value(self):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        R = set(self.preds)
        T = set(self.target)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return {'f1': f1, "precision": precision, "recall": recall}

    def name(self):
        return 'gp'


class SequenceLabelingTrainer(TrainBaseBuilder):
    keys_to_ignore_on_checkpoint_save = ['optimizer']  # checkpoint中不存储的模块，比如'optimizer'

    def process_batch_outputs(self, outputs):
        pred_entities = []
        true_entities = []
        labels = concat_tensors_with_padding(outputs['logits'], padding_index=-1, padding_shape=[0, 1, 0, 1],
                                             padding_value=0)
        logits = concat_tensors_with_padding(outputs['labels'], padding_index=-1, padding_shape=[0, 1, 0, 1],
                                             padding_value=0)
        y_pred = tensor_to_numpy(logits)
        y_true = tensor_to_numpy(labels)
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred_entities.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true_entities.append((b, l, start, end))
        return {"preds": pred_entities, "target": true_entities}


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            ResumeDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return ResumeDataset(data_name, data_dir, data_type, process_piplines, **kwargs)


MODEL_CLASSES = {
    "bert": (BertConfig, BertGlobalPointerForSeqLabel, BertTokenizer),
}


def main():
    parser = Argparser.build_parser()
    group = parser.add_argument_group(title="global pointer", description="Global pointer")
    group.add_argument('--use_rope', action='store_true')
    group.add_argument('--inner_dim', default=64, type=int, help='The dim of Positional embedding')
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
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    config.use_rope = opts.use_rope
    config.inner_dim = opts.inner_dim
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = SequenceLabelingTrainer(opts=opts,
                                      model=model,
                                      metrics=[GobalPointerMetric()],
                                      logger=logger
                                      )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
