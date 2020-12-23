import os
import json
import torch
from torchblocks.metrics import SequenceLabelingScore
from torchblocks.trainer import SequenceLabelingTrainer
from torchblocks.callback import TrainLogger
from torchblocks.processor import SequenceLabelingProcessor, InputExample
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from torchblocks.data import CNTokenizer
from torchblocks.models.nn import BertCRFForAttr
from transformers import WEIGHTS_NAME, BertConfig

MODEL_CLASSES = {
    'bert': (BertConfig, BertCRFForAttr, CNTokenizer)
}


class AttrProcessor(SequenceLabelingProcessor):

    def get_labels(self):
        return ["X", "B-a", "I-a", 'S-a', 'O', "[START]", "[END]"]

    def collate_fn(self, batch):
        """
        batch should be a list of (input_ids, attention_mask, *,*,*, labels) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        batch = list(map(torch.stack, zip(*batch)))
        max_seq_len = torch.max(torch.sum(batch[1], 1)).item()  # title
        max_attr_len = torch.max(torch.sum(batch[4], 1)).item()  # attribute
        for i in [0, 1, 2, 6]:
            if batch[i].size()[1] > max_seq_len:
                batch[i] = batch[i][:, :max_seq_len]
        for i in [3, 4, 5]:
            if batch[i].size()[1] > max_attr_len:
                batch[i] = batch[i][:, :max_attr_len]
        return batch

    def read_data(self, input_file):
        """Reads a json list file."""
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                title = line['title']
                attr = line['attribute']
                value = line['value']
                labels = ['O'] * len(title)
                if value != '':
                    assert value in title
                    s = title.find(value)
                    if len(value) == 1:
                        labels[s] = 'S-a'
                    else:
                        labels[s] = 'B-a'
                        labels[s + 1:s + len(value)] = ['I-a'] * (len(value) - 1)
                assert len(labels) == len(title)
                lines.append({"title": title, "labels": labels, 'attr': attr})
        return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            title = line['title']
            attribute = line['attr']
            labels = line['labels']
            # 标签序列使用label_ids
            examples.append(InputExample(guid=guid, texts=[[title, None], [attribute, None]], label_ids=labels))
        return examples


def main():
    parser = build_argparse()
    parser.add_argument('--markup', type=str, default='bios', choices=['bios', 'bio'])
    parser.add_argument('--max_attr_length', default=16, type=int)
    args = parser.parse_args()

    # output dir
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
    args.output_dir = args.output_dir + '{}'.format(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
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
    processor = AttrProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix, encode_mode='pair')
    label_list = processor.get_labels()
    num_labels = len(label_list)
    id2label = {i: label for i, label in enumerate(label_list)}
    args.id2label = id2label
    args.num_labels = num_labels

    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path, num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    # Trainer
    logger.info("initializing traniner")
    trainer = SequenceLabelingTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                                      input_keys=processor.get_input_keys(),
                                      metrics=[SequenceLabelingScore(id2label, markup=args.markup)])
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
