import os
import pandas as pd
from torchblocks.metrics import AUC, MultiLabelReport
from torchblocks.trainer import TextClassifierTrainer
from torchblocks.callback import TrainLogger
from torchblocks.processor import TextClassifierProcessor, InputExample
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from torchblocks.models.nn import BertForMultiLable
from transformers import BertConfig, BertTokenizer
from transformers import WEIGHTS_NAME

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultiLable, BertTokenizer)
}


class ToxicProcessor(TextClassifierProcessor):

    def get_labels(self):
        """See base class."""
        return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def read_data(self, input_file):
        """Reads a json list file."""
        lines = pd.read_csv(input_file)
        return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(lines.values):
            guid = row[0]
            text_a = row[1]
            label_ids = row[2:] if set_type != 'test' else None
            examples.append(
                InputExample(guid=guid, texts=[text_a, None], label_ids=label_ids))
        return examples


def main():
    args = build_argparse().parse_args()
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]

    # output dir
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
    processor = ToxicProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix)
    label_list = processor.get_labels()
    id2label = {i: key for i, key in enumerate(label_list)}
    num_labels = len(label_list)
    args.num_labels = num_labels

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    # Trainer
    logger.info("initializing traniner")
    trainer = TextClassifierTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                                    batch_input_keys=processor.get_batch_keys(),
                                    metrics=[AUC(average='micro', task_type='binary'),
                                             MultiLabelReport(id2label)])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.csv', 'train')
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.csv', 'dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    # do eval
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.csv', 'dev')
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
        test_dataset = processor.create_dataset(args.eval_max_seq_length, 'test.csv', 'test')
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
