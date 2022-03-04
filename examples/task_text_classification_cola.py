import os
import csv
from typing import List, Dict, Callable, Any
from torchblocks.core import TextClassifierTrainer
from torchblocks.data.dataset import DatasetBase
from torchblocks.utils.seed import seed_everything
from torchblocks.utils.options import Argparser
from torchblocks.utils.device import prepare_device
from torchblocks.utils.logger import Logger
from torchblocks.utils.paths import check_dir
from torchblocks.data.process_base import ProcessBase
from torchblocks.utils.paths import find_all_checkpoints
from torchblocks.metrics.classification.matthews_corrcoef import MattewsCorrcoef
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


class ColaDataset(DatasetBase):

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines: List[Callable],
                 **kwargs):
        super().__init__(data_name, data_dir, data_type, process_piplines, **kwargs)

    @classmethod
    def get_labels(self) -> List[str]:
        return ["0", "1"]

    def read_data(self, input_file: str) -> Any:
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t"))

    def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
        test_mode = data_type == "test"
        if test_mode:
            data = data[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            text = line[text_index]
            label = None if test_mode else line[1]
            examples.append(dict(guid=guid, text=text, label=label))
        return examples


class ProcessEncodeText(ProcessBase):
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, tokenizer, tokenizer_params, return_input_length=False):
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        self.return_input_length = return_input_length

    def __call__(self, example):
        inputs = self.tokenizer(example["text"], **self.tokenizer_params)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        if self.return_input_length:
            inputs["input_length"] = inputs["attention_mask"].sum().item()
        example = dict(example, **inputs)
        return example


class ProcessEncodeLabel(ProcessBase):
    """ 编码单标签文本标签 """

    def __init__(self, label2id):
        self.label2id = label2id

    def __call__(self, example):
        example["label"] = self.label2id.get(example["label"], None)
        return example


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length):
    process_piplines = [ProcessEncodeText(tokenizer,
                                          tokenizer_params={
                                              "padding": "max_length",
                                              "truncation": "longest_first",
                                              "max_length": max_sequence_length,
                                              "return_tensors": "pt",
                                          }),
                        ProcessEncodeLabel(ColaDataset.label2id())
                        ]
    return ColaDataset(data_name=data_name,
                       data_dir=data_dir,
                       data_type=data_type,
                       process_piplines=process_piplines
                       )


def main():
    opts = Argparser().get_training_arguments()
    logger = Logger(opts=opts)
    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    train_dataset = load_data(opts.train_input_file, opts.data_dir, "train", tokenizer, opts.train_max_seq_length)
    dev_dataset = load_data(opts.eval_input_file, opts.data_dir, "dev", tokenizer, opts.eval_max_seq_length)
    opts.num_labels = train_dataset.num_labels
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = TextClassifierTrainer(opts=opts,
                                    model=model,
                                    metrics=[MattewsCorrcoef(num_classes=opts.num_labels)],
                                    logger=logger
                                    )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})
    if opts.do_eval:
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        if opts.eval_all_checkpoints:
            checkpoints = find_all_checkpoints(checkpoint_dir=opts.output_dir)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.evaluate(dev_data=dev_dataset, save_result=True, save_dir=prefix)

    if opts.do_predict:
        test_dataset = load_data(opts.test_input_file, opts.data_dir, "test", tokenizer, opts.test_max_seq_length)
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.predict(test_data=test_dataset, save_result=True, save_dir=prefix)


if __name__ == "__main__":
    main()
