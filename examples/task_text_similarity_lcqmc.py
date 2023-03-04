import torch
import csv
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils import seed_everything
from torchblocks.utils.options import Argparser
from torchblocks.utils.device import build_device
from torchblocks.utils.logger import Logger
from torchblocks.metrics.classification.accuracy import Accuracy
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer, BertModel


class BertForSequenceClassification(BertPreTrainedModel, Application):
    def __init__(self, config):
        super().__init__(config)
        self.initializer_range = config.initializer_range
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def compute_loss(self, logits, labels):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        labels = inputs.get("labels", None)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}


class LcqmcDataset(DatasetBaseBuilder):

    @staticmethod
    def get_labels():
        return ["0", "1"]

    def read_data(self, input_file):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def build_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            text_a = line[0]
            text_b = line[1]
            label = str(int(line[2])) if data_type != 'test' else None
            examples.append(dict(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ProcessEncodeText:
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, tokenizer,tokenizer_params, return_input_length=False):
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        self.return_input_length = return_input_length

    def __call__(self, example):
        inputs = self.tokenizer.encode_plus(text=example["text_a"], text_pair=example['text_b'],
                                            **self.tokenizer_params)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        if example['label'] is not None:
            inputs["label"] = example["label"]
        return inputs


class ProcessEncodeLabel:
    """ 编码单标签文本标签 """

    def __init__(self, label2id):
        self.label2id = label2id

    def __call__(self, example):
        example["label"] = self.label2id.get(example["label"], None)
        return example


def load_data(opts, file_name, data_type, tokenizer, max_sequence_length):
    process_piplines = [ProcessEncodeText(tokenizer,
                                          tokenizer_params={
                                              "padding": "max_length",
                                              "truncation": "longest_first",
                                              "max_length": max_sequence_length,
                                              "return_tensors": "pt",
                                          }),
                        ProcessEncodeLabel(LcqmcDataset.label2id())
                        ]
    return LcqmcDataset(opts, file_name, data_type=data_type, process_piplines=process_piplines)


class TextClassifierTrainer(TrainBaseBuilder):
    '''
    文本分类
    '''

    # 跟model的输出、metric的输入相关
    def process_batch_outputs(self, batches, dim=0):
        preds = torch.cat([batch for batch in batches['logits']], dim=dim)
        target = torch.cat([batch for batch in batches['labels']], dim=dim)
        return {"preds": preds, "target": target}


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
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
    opts.num_labels = len(train_dataset.get_labels())
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = TextClassifierTrainer(opts=opts,
                                    model=model,
                                    metrics=Accuracy(task="multiclass",num_classes=opts.num_labels),
                                    logger=logger
                                    )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
