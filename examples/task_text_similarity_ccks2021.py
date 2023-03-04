import json
import torch
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
        self.high_dropout = nn.Dropout(p=0.5)
        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def compute_loss(self, logits, labels):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def forward(self, inputs):
        input_ids =inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs.get("labels",None)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_layers = outputs[2]
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        loss = self.compute_loss(logits, labels)
        outputs = {"loss": loss, "logits": logits}
        return outputs


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


class CCKSDataset(DatasetBaseBuilder):

    @staticmethod
    def get_labels():
        return {'不匹配': 0, '部分匹配': 1, '完全匹配': 2}

    def read_data(self, input_file):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f.readlines():
                lines.append(json.loads(line))
            return lines

    def build_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            text_a = line['query']
            text_b = line['candidate']
            label = line['label'] if data_type != 'test' else None
            examples.append(dict(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ProcessEncodeText:
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, label2id, tokenizer, tokenizer_params, return_input_length=False):
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        self.return_input_length = return_input_length

    def __call__(self, example):
        encoding = self.tokenizer.encode_plus(text=example["text_a"],
                                              text_pair=example['text_b'],
                                              **self.tokenizer_params)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["label"] = self.label2id.get(example["label"], None)
        return encoding


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
    process_piplines = [ProcessEncodeText(CCKSDataset.label2id(),
                                          tokenizer,
                                          tokenizer_params={
                                              "padding": "max_length",
                                              "truncation": "longest_first",
                                              "max_length": max_sequence_length,
                                              "return_tensors": "pt",
                                          })
                        ]
    return CCKSDataset(opts, file_name, data_type=data_type, process_piplines=process_piplines)


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
    config = config_class.from_pretrained(opts.pretrained_model_path)
    config.update(
        {
            "output_hidden_states": True,
            "hidden_dropout_prob": 0.1,
            "layer_norm_eps": 1e-7,
            "add_pooling_layer": False,
            "num_labels": opts.num_labels,
        }
    )
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
