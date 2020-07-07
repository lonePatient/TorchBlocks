import os

from torchblocks.metrics import NERScore
from torchblocks.trainer import SequenceLabelingSpanTrainer
from torchblocks.callback import TrainLogger
from torchblocks.processor import SequenceLabelingSpanProcessor, InputExample
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from torchblocks.data import CNTokenizer
from torchblocks.models.nn import BertSpanForNer
from transformers import WEIGHTS_NAME, BertConfig
from torchblocks.metrics.ner_utils import get_spans

MODEL_CLASSES = {
    'bert': (BertConfig, BertSpanForNer, CNTokenizer)
}


class CnerProcessor(SequenceLabelingSpanProcessor):
    def __init__(self, markup, tokenizer, data_dir, prefix=''):
        super().__init__(tokenizer=tokenizer, data_dir=data_dir, prefix=prefix)
        self.markup = markup

    def get_labels(self):
        """See base class."""
        return ["O", "CONT", "ORG", "LOC", 'EDU', 'NAME', 'PRO', 'RACE', 'TITLE']

    def read_data(self, input_file):
        """Reads a json list file."""
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": "".join(words), "labels": labels})
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
                lines.append({"words": "".join(words), "labels": labels})
        return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            # 标签序列使用label_ids
            spans = get_spans(labels, id2label=None, markup=self.markup)
            examples.append(InputExample(guid=guid, texts=[text_a, None], label_ids=spans))
        return examples


def main():
    parser = build_argparse()
    parser.add_argument('--markup', type=str, default='bios', choices=['bios', 'bio'])
    args = parser.parse_args()
    # output dir
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
    args.output_dir = args.output_dir + '{}'.format(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = "_".join([args.model_name, args.task_name])
    logger = TrainLogger(log_dir=args.output_dir, prefix=prefix)
    # device
    logger.info("initializing device")
    args.device, args.n_gpu = prepare_device(args.gpu, args.local_rank)
    seed_everything(args.seed)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    processor = CnerProcessor(args.markup, tokenizer, args.data_dir, prefix=prefix)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    id2label = {i: label for i, label in enumerate(label_list)}
    args.id2label = id2label
    args.num_labels = num_labels
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    # trainer
    logger.info("initializing traniner")
    trainer = SequenceLabelingSpanTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                                          batch_input_keys=processor.get_batch_keys(),
                                          metrics=[NERScore(id2label, markup=args.markup, is_spans=True)])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.char.bmes', 'train')
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.char.bmes', 'dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    # do eval
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.char.bmes', 'dev')
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints or args.checkpoint_number > 0:
            checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("/")[-1].split("-")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            trainer.evaluate(model, eval_dataset, save_preds=True, prefix=str(global_step))
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in trainer.records['result'].items()}
                results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        dict_to_text(output_eval_file, results)
    # do predict
    if args.do_predict:
        test_dataset = processor.create_dataset(args.eval_max_seq_length, 'test.char.bmes', 'test')
        if args.checkpoint_number == 0:
            raise ValueError("checkpoint number should > 0,but get %d", args.checkpoint_number)
        checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("/")[-1].split("-")[-1]
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            trainer.predict(model, test_dataset=test_dataset, prefix=str(global_step))


if __name__ == "__main__":
    main()
