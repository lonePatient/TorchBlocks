import json
import torch
import torch.nn as nn
from torchblocks.core import TrainBaseBuilder, Application
from torchblocks.data import DatasetBaseBuilder
from torchblocks.utils import seed_everything
from torchblocks.utils.options import Argparser
from torchblocks.utils.device import build_device
from torchblocks.utils.logger import Logger
from torchblocks.metrics.classification.matthews_corrcoef import MattewsCorrcoef
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer, BertModel

class BertForSequenceClassification(BertPreTrainedModel, Application):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def compute_loss(self, outputs, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
        return loss

    def forward(self, inputs):
        outputs = self.bert(inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type_ids'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        labels = inputs.get("labels", None)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}


# 定义数据集加载
class TnewsDataset(DatasetBaseBuilder):

    @staticmethod
    def get_labels():
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def read_data(self, input_file: str):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f.readlines():
                lines.append(json.loads(line))
            return lines

    def build_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            text = line['sentence']
            label = line['label'] if data_type != 'test' else None
            examples.append(dict(guid=guid, text=text, label=label))
        return examples

# 数据的处理
class ProcessEncodeText:
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, tokenizer, label2id, tokenizer_params, return_input_length=False):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.tokenizer_params = tokenizer_params
        self.return_input_length = return_input_length

    def __call__(self, example):
        inputs = self.tokenizer(example["text"], **self.tokenizer_params)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        if example['label'] is not None:
            inputs["label"] = self.label2id[example["label"]]
        return inputs


# 定义任务的训练模块
class TextClassifierTrainer(TrainBaseBuilder):
    '''
    文本分类
    '''
    # 跟model的输出、metric的输入相关
    def process_batch_outputs(self, batches, dim=0):
        preds = torch.cat([batch for batch in batches['logits']], dim=dim)
        target = torch.cat([batch for batch in batches['labels']], dim=dim)
        return {"preds": preds, "target": target}


def load_data(opts, file_name, data_type, tokenizer, max_sequence_length):
    process_piplines = [ProcessEncodeText(tokenizer, TnewsDataset.label2id(),
                                          tokenizer_params={
                                              "padding": "max_length",
                                              "truncation": "longest_first",
                                              "max_length": max_sequence_length,
                                              "return_tensors": "pt",
                                          })
                        ]
    return TnewsDataset(opts, file_name, data_type, process_piplines)


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
    opts.num_labels = len(train_dataset.label2id())

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    config.output_hidden_states = False
    config.hidden_dropout_prob = 0.
    config.attention_probs_dropout_prob = 0.
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = TextClassifierTrainer(opts=opts,
                                    model=model,
                                    metrics=[MattewsCorrcoef(task="multiclass",num_classes=opts.num_labels)],
                                    logger=logger
                                    )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})


if __name__ == "__main__":
    main()
