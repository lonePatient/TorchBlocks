import os
import torch
import json
from torchblocks.metrics import NERScore
from torchblocks.trainer import SequenceLabelingTrainer
from torchblocks.callback import ModelCheckpoint, TrainLogger
from torchblocks.processor import SequenceLabelingProcessor, InputExample, InputFeatures
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from torchblocks.processor import CNTokenizer
from torchblocks.models.nn import BertCrfForAttr
from transformers import WEIGHTS_NAME, BertConfig

MODEL_CLASSES = {
    'bert': (BertConfig, BertCrfForAttr, CNTokenizer)
}


class AttrProcessor(SequenceLabelingProcessor):
    def __init__(self, tokenizer, data_dir, logger, prefix=''):
        super().__init__(tokenizer=tokenizer, data_dir=data_dir, logger=logger, prefix=prefix)

    def get_labels(self):
        return ["X", "B-a", "I-a", 'S-a', 'O', "[START]", "[END]"]

    def get_batch_keys(self):
        return ['input_ids', 'attention_mask', 'token_type_ids',
                'a_input_ids', 'a_attention_mask', 'a_token_type_ids',
                'label_ids']

    def collate_fn(self, batch):

        """
        batch should be a list of (input_ids, attention_mask, *,*,*, labels) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        batch = list(map(torch.stack, zip(*batch)))
        max_seq_len = torch.max(torch.sum(batch[1], 1)).item()
        max_attr_len = torch.max(torch.sum(batch[4], 1)).item()
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

    def convert_to_features(self, examples, label_list, max_seq_length, max_attr_length):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                self.logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            inputs = self.tokenizer.encode_plus(text=example.texts[0],
                                                text_pair=None,
                                                max_length=max_seq_length,
                                                add_special_tokens=self.add_special_tokens,
                                                pad_to_max_length=self.pad_to_max_length
                                                )
            attr_inputs = self.tokenizer.encode_plus(text=example.texts[1],
                                                     text_pair=None,
                                                     max_length=max_attr_length,
                                                     add_special_tokens=self.add_special_tokens,
                                                     pad_to_max_length=self.pad_to_max_length
                                                     )
            inputs.update({'a_' + key: value for key, value in attr_inputs.items()})
            # label
            label_ids = example.label_ids
            special_toekns_num = 2 if self.add_special_tokens else 0
            if len(label_ids) > max_seq_length - special_toekns_num:  # [CLS] and [SEP]
                label_ids = label_ids[:(max_seq_length - special_toekns_num)]
            label_ids = [label_map[x] for x in label_ids]
            label_ids = [label_map[self.special_token_label]] + label_ids + [label_map[self.special_token_label]]
            label_ids += [self.pad_label_id] * (max_seq_length - len(label_ids))

            inputs['guid'] = example.guid
            inputs['label_ids'] = label_ids
            if ex_index < 5:
                self.print_examples(**inputs)
            features.append(InputFeatures(**inputs))
        return features


def main():
    parser = build_argparse()
    parser.add_argument('--markup', type=str, default='bios', choices=['bios', 'bio'])
    parser.add_argument('--max_attr_length', default=16, type=int)
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
    args.output_dir = args.output_dir + '{}'.format(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    prefix = "_".join([args.model_name, args.task_name])
    logger = TrainLogger(log_dir=args.output_dir, prefix=prefix)

    logger.info("initializing device")
    args.device, args.n_gpu = prepare_device(args.gpu, args.local_rank)

    seed_everything(args.seed)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    processor = AttrProcessor(tokenizer, args.data_dir, logger, prefix=prefix)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    id2label = {i: label for i, label in enumerate(label_list)}
    args.id2label = id2label
    args.num_labels = num_labels

    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)


    logger.info("initializing traniner")
    trainer = SequenceLabelingTrainer(logger=logger,
                                      args=args,
                                      batch_input_keys=processor.get_batch_keys(),
                                      collate_fn=processor.collate_fn,
                                      metrics=[NERScore(id2label, markup=args.markup)])
    if args.do_train:
        train_dataset = processor.create_dataset(max_seq_length=args.train_max_seq_length,
                                                 data_name='train.char.bmes', mode='train')
        eval_dataset = processor.create_dataset(max_seq_length=args.eval_max_seq_length,
                                                data_name='dev.char.bmes', mode='dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)

    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = processor.create_dataset(max_seq_length=args.eval_max_seq_length,
                                                data_name='dev.char.bmes', mode='dev')
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

    if args.do_predict:
        test_dataset = processor.create_dataset(max_seq_length=args.eval_max_seq_length,
                                                data_name='test.char.bmes', mode='test')
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
