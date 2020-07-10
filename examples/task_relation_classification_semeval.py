import os
import csv
from torchblocks.metrics import F1Score, Accuracy
from torchblocks.trainer import TextClassifierTrainer
from torchblocks.callback import TrainLogger
from torchblocks.models.nn import REBERT
from torchblocks.processor import DataProcessor, InputExample, InputFeatures
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from transformers import BertConfig, BertTokenizer, WEIGHTS_NAME

MODEL_CLASSES = {
    'bert': (BertConfig, REBERT, BertTokenizer)
}

'''
Enriching Pre-trained Language Model with Entity Information for Relation Classification
'''


class SemEvalProcessor(DataProcessor):

    def get_labels(self):
        """See base class."""
        return [label.strip() for label in open(self.label_path, 'r', encoding='utf-8')]

    def read_data(self, input_file):
        """Reads a json list file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, texts=[text_a, None], label=label))
        return examples

    def get_batch_keys(self):
        return ['input_ids', 'attention_mask', 'token_type_ids', 'e1_mask', 'e2_mask', 'labels']

    def convert_to_features(self, examples, label_list, max_seq_length):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                print("Writing example %d/%d" % (ex_index, len(examples)))
            texts = example.texts
            tokens_a = self.tokenizer.tokenize(texts[0])
            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2
            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"
            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2 if self.add_sep_token else 1
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
            if self.add_sep_token:
                tokens_a += ['[SEP]']
            tokens_a = ['[CLS]'] + tokens_a
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens_a)
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            padding_tokens = [0] * padding_length
            # Zero-pad up to the sequence length.
            input_ids = input_ids + padding_tokens
            attention_mask = attention_mask + padding_tokens
            token_type_ids = token_type_ids + padding_tokens
            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)
            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1
            label = int(label_map[example.label])
            inputs = {}
            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_mask
            inputs['token_type_ids'] = token_type_ids
            inputs['e2_mask'] = e2_mask
            inputs['e1_mask'] = e1_mask
            inputs['label'] = label
            if ex_index < 5:
                self.print_examples(**inputs)
            features.append(InputFeatures(**inputs))
        return features


def main():
    parser = build_argparse()
    parser.add_argument('--label_file', type=str, default='label.txt')
    # output dir
    args = parser.parse_args()
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
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
    processor = SemEvalProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix, add_sep_token=False,
                                 label_path=os.path.join(args.data_dir, args.label_file))
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path, num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    # Trainer
    logger.info("initializing traniner")
    trainer = TextClassifierTrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                                    batch_input_keys=processor.get_batch_keys(),
                                    metrics=[F1Score(average='macro', task_type='multiclass'), Accuracy()])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.tsv', 'train')
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.tsv', 'dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    # do eval
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.tsv', 'dev')
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
        test_dataset = processor.create_dataset(args.eval_max_seq_length, 'test.tsv', 'test')
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
