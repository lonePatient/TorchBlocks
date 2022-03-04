import os
import torch
from typing import *
from transformers import (
    BertConfig,
    BertTokenizer,
)
from torchblocks.data.dataset import DatasetBase
from torchblocks.data.process_base import ProcessBase
from torchblocks.tasks.sequence_labeling_softmax import BertSoftmaxForSeqLabel
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.core import SequenceLabelingTrainer
from torchblocks.utils.device import prepare_device
from torchblocks.utils.paths import check_dir
from torchblocks.utils.paths import find_all_checkpoints
from torchblocks.utils.seed import seed_everything


class CnerDataset(DatasetBase):
    keys_to_truncate_on_dynamic_batch = [
        'input_ids', 'attention_mask', 'token_type_ids', 'labels'
    ]

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines: List[Callable],
                 **kwargs):
        super().__init__(data_name, data_dir, data_type, process_piplines, **kwargs)

    @classmethod
    def get_labels(self) -> List[str]:
        labels = ["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                  'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                  'O', 'S-NAME', 'S-ORG', 'S-RACE']
        return labels

    def read_data(self, input_file: str) -> Any:
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append([words, labels])
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append([words, labels])
        return lines

    def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
        examples = []
        for (i, line) in enumerate(data):
            guid = f"{data_type}-{i}"
            tokens = line[0]
            labels = []
            for x in line[1]:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(dict(guid=guid, tokens=tokens, labels=labels))
        return examples


class ProcessExample2Feature(ProcessBase):

    def __init__(self, label2id, tokenizer, max_sequence_length):
        super().__init__()
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, example):
        tokens = example['tokens']
        labels = example['labels']

        inputs = self.tokenizer(
            tokens,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_sequence_length,
            return_overflowing_tokens=True,
            is_split_into_words=True,
            return_tensors='pt',
        )
        overflowing_tokens = inputs.pop("overflowing_tokens")
        num_truncated_tokens = inputs.pop("num_truncated_tokens")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        if labels is None:
            inputs['label_ids'] = None
            return inputs

        truncate_len = len(tokens) - overflowing_tokens.size(-1)
        labels = ['O'] + labels[: truncate_len] + ['O']
        labels = labels + ['O'] * (self.max_sequence_length - truncate_len - 2)
        label_ids = [self.label2id[label] for label in labels]
        inputs['label_ids'] = torch.tensor(label_ids)
        return inputs


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length, **kwargs):
    process_piplines = [
        ProcessExample2Feature(
            CnerDataset.label2id(), tokenizer, max_sequence_length),
    ]
    return CnerDataset(data_name, data_dir, data_type, process_piplines, **kwargs)


MODEL_CLASSES = {
    "bert": (BertConfig, BertSoftmaxForSeqLabel, BertTokenizer),
}


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
    test_dataset = load_data(opts.test_input_file, opts.data_dir, "test", tokenizer, opts.test_max_seq_length)
    opts.num_labels = train_dataset.num_labels
    opts.label2id = CnerDataset.label2id()
    opts.id2label = CnerDataset.id2label()

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path,
                                          num_labels=opts.num_labels, label2id=opts.label2id, id2label=opts.id2label)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)

    # trainer
    logger.info("initializing traniner")
    labels = {label.split('-')[1] for label in CnerDataset.get_labels() if '-' in label}
    metrics = [SequenceLabelingScore(labels=labels, average='micro', schema='BIOS')]
    trainer = SequenceLabelingTrainer(opts=opts,
                                      model=model,
                                      metrics=metrics,
                                      logger=logger)
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
