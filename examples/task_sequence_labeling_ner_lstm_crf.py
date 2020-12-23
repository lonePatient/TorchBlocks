import os
import json
from torchblocks.metrics import SequenceLabelingScore
from torchblocks.trainer import SequenceLabelingTrainer
from torchblocks.callback import TrainLogger
from torchblocks.processor import SequenceLabelingProcessor, InputExample
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from torchblocks.data import CNTokenizer
from torchblocks.data import Vocabulary, VOCAB_NAME
from torchblocks.models.nn.lstm_crf import LSTMCRF
from torchblocks.models.bases import TrainConfig
from torchblocks.models.bases import WEIGHTS_NAME

MODEL_CLASSES = {
    'lstm-crf': (TrainConfig, LSTMCRF, CNTokenizer)
}


def build_vocab(data_dir, vocab_dir):
    '''
    构建vocab
    '''
    vocab = Vocabulary()
    vocab_path = os.path.join(vocab_dir, VOCAB_NAME)
    if os.path.exists(vocab_path):
        vocab.load_vocab(str(vocab_path))
    else:
        files = ["train.json", "dev.json", "test.json"]
        for file in files:
            with open(os.path.join(data_dir, file), 'r') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    text = line['text']
                    vocab.update(list(text))
        vocab.build_vocab()
        vocab.save_vocab(vocab_path)
    print("vocab size: ", len(vocab))


class CluenerProcessor(SequenceLabelingProcessor):

    def get_labels(self):
        """See base class."""
        # 默认第一个为X
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position', 'B-scene', "I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position', 'I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene', 'O', "[START]", "[END]"]

    def read_data(self, input_file):
        """Reads a json list file."""
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                labels = ['O'] * len(text)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert text[start_index:end_index + 1] == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                lines.append({"text": text, "labels": labels})
        return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text']
            labels = line['labels']
            examples.append(InputExample(guid=guid, texts=[text_a, None], label_ids=labels))
        return examples


def main():
    parser = build_argparse()
    parser.add_argument('--markup', type=str, default='bios', choices=['bios', 'bio'])
    parser.add_argument('--use_crf', action='store_true', default=True)
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
    # build vocab
    build_vocab(args.data_dir, vocab_dir=args.model_path)
    seed_everything(args.seed)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    processor = CluenerProcessor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix,add_special_tokens=False)
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
    trainer = SequenceLabelingTrainer(args=args, logger=logger, collate_fn=processor.collate_fn,
                                      input_keys=processor.get_input_keys(),
                                      metrics=[SequenceLabelingScore(id2label, markup=args.markup)])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.json', 'train', )
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.json', 'dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    # do eval
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.json', 'dev')
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
        test_dataset = processor.create_dataset(args.eval_max_seq_length, 'test.json', 'test')
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
