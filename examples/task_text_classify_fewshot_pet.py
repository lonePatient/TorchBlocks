import torch
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


class BertForMaskedLM(BertPreTrainedModel, Application):
    # _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def compute_loss(self, outputs, labels, **kwargs):
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


# 定义数据集加载
class FewShotPETDataset(DatasetBaseBuilder):
    # collecate
    keys_to_ignore_on_collate_batch = ['raw_label', 'mask_span_indices']
    # 动态batch处理过程中需要进行按照batch最长长度进行截取的keys
    keys_to_dynamical_truncate_on_padding_batch = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']

    @staticmethod
    def get_labels():
        return ["0", "1"]

    @staticmethod
    def get_label_desc():
        return ["否", "能"]

    def read_data(self, input_file):
        with open(input_file, 'r') as f:
            data_rows = f.readlines()
        return data_rows

    def build_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            lines = line.strip("\n").split("\t")
            guid = f"{data_type}-{i}"
            text_a = lines[1]
            text_b = lines[2]
            label = lines[3] if data_type != 'test' else None
            examples.append(dict(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# 数据的处理
class ProcessEncodeText:
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, tokenizer, max_sequence_length, label2desc):
        self.tokenizer = tokenizer
        self.label2desc = label2desc
        self.max_sequence_length = max_sequence_length
        self.vocab_size = len(self.tokenizer.vocab)
        self.pad_idx = self.tokenizer.pad_token_id
        self.label_length = len(tokenizer.tokenize(list(label2desc.values())[0]))  # label的长度对应mask个数

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __call__(self, example):
        # pattern: sent1,label,用,sent2,概括。
        text_a = example['text_a']
        text_b = example['text_b']
        raw_label = example['label']
        num_extra_tokens = self.label_length + 3  # num_extra_tokens:能/否 用 概 括。
        tokens_a = self.tokenizer.tokenize(text_a)
        max_seq_length = self.max_sequence_length - num_extra_tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)  # cls sep

        # 构建pattern: sent1,label,用,sent2,概括。
        tokens = [self.tokenizer.cls_token]  # cls
        tokens += tokens_a  # text_a
        label_position = len(tokens)
        tokens += [self.tokenizer.mask_token] * self.label_length  # [MASK]插入
        tokens += self.tokenizer.tokenize("用")  #
        tokens += tokens_b  # text_b
        tokens += self.tokenizer.tokenize("概括")
        tokens += [self.tokenizer.sep_token]  # sep
        # 转化
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        length = len(input_ids)
        attention_mask = [1] * length
        token_type_ids = [0] * length
        # token padding
        padding_length = self.max_sequence_length - length
        input_ids += [self.pad_idx] * padding_length
        attention_mask += [self.pad_idx] * padding_length
        token_type_ids += [0] * padding_length
        mask_span_indices = []
        for i in range(self.label_length):
            mask_span_indices.append([label_position + i])
        mask_labels = None
        if raw_label is not None:
            label_desc = self.label2desc[raw_label]
            label_desc_tokens = self.tokenizer.tokenize(label_desc)
            label_tokens_ids = self.tokenizer.convert_tokens_to_ids(label_desc_tokens)
            mask_labels = [-100] * self.max_sequence_length
            for i in range(self.label_length):
                mask_labels[label_position + i] = label_tokens_ids[i]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'label_ids': torch.tensor(mask_labels),
            'raw_label': int(raw_label),
            'mask_span_indices': mask_span_indices
        }


# 定义任务的训练模块
class FewShotPETTrainer(TrainBaseBuilder):
    '''
    文本分类
    '''
    keys_to_ignore_on_gpu = ['raw_label', 'mask_span_indices']  # batch数据中不转换为GPU的变量名
    keys_to_ignore_on_save_result = ['input_ids', 'token_type_ids']  # eval和predict结果不存储的变量
    keys_to_ignore_on_save_checkpoint = ["optimizer"]  # checkpoint中不存储的模块，比如'optimizer'

    # 跟model的输出、metric的输入相关
    def process_batch_outputs(self, batches, dim=0):
        batch_num = len(batches['logits'])
        desc2ids = self.opts.desc2ids
        label2desc = self.opts.label2desc
        desc2label = {value: key for key, value in label2desc.items()}
        target = []
        preds = []
        # 处理desc到真实label的映射
        for b in range(batch_num):
            logits = batches['logits'][b].float().log_softmax(dim=-1)
            mask_span_indices = batches['mask_span_indices'][b]
            raw_labels = batches['raw_label'][b]
            for i in range(logits.shape[0]):
                y_logits = logits[i]
                indices = mask_span_indices[i]
                target.append(raw_labels[i])
                pred_label_probs = []
                # 计算预测标签prob
                for key, value in desc2ids.items():
                    pred_prob = 0.
                    # subword采用相加方式
                    for l_ids, span_indices in zip(value, indices):
                        span_idx = span_indices[0]
                        pred_prob += y_logits[span_idx, l_ids]
                    pred_label_probs.append([key, pred_prob])
                pred_label = sorted(pred_label_probs, key=lambda x: x[1], reverse=True)[0][0]
                preds.append(int(desc2label[pred_label]))
        return {'preds': torch.tensor(preds), 'target': torch.tensor(target)}


def load_data(opts, file_name, data_type, tokenizer, max_sequence_length):
    process_piplines = [ProcessEncodeText(tokenizer, max_sequence_length, opts.label2desc)]
    return FewShotPETDataset(opts, file_name, data_type, process_piplines)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer)
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
    labels = FewShotPETDataset.get_labels()
    label_desc = FewShotPETDataset.get_label_desc()
    label2desc = dict(zip(labels, label_desc))  # 原始标签与desc的对应关系
    desc2ids = {key: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(key)) for key in label_desc}
    opts.desc2ids = desc2ids  # 每一个label desc对应的tokenizer的ids，主要用于eval和test过程
    opts.label2desc = label2desc
    opts.num_labels = len(labels)
    train_dataset = load_data(opts, opts.train_input_file, "train", tokenizer, opts.train_max_seq_length)
    dev_dataset = load_data(opts, opts.eval_input_file, "dev", tokenizer, opts.eval_max_seq_length)

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = FewShotPETTrainer(opts=opts, model=model,
                                metrics=[Accuracy(task="multiclass", num_classes=opts.num_labels)], logger=logger)
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer},
                      convert_output_cuda_to_cpu=True)


if __name__ == "__main__":
    main()
