import os
import csv
import torch
import torch.nn as nn
from copy import deepcopy
from torchblocks.metrics import Accuracy
from torchblocks.callback import TrainLogger
from torchblocks.trainer import TextClassifierTrainer
from torchblocks.processor import TextClassifierProcessor, InputExample
from torchblocks.utils import seed_everything, build_argparse
from torchblocks.utils import prepare_device
from transformers import BertConfig, BertTokenizer
from torchblocks.models.transformer import BertForSequenceClassification
from torchblocks.models.transformer.utils import ConstantReplacementScheduler, LinearReplacementScheduler

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


class Sst2Processor(TextClassifierProcessor):

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def read_data(self, input_file):
        """Reads a json list file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, texts=[text_a, None], label=label))
        return examples


class TheseusTrainer(TextClassifierTrainer):
    def __init__(self, args, metrics, logger, batch_input_keys, replacing_rate_scheduler, collate_fn=None):
        super().__init__(args=args, metrics=metrics, logger=logger,
                         batch_input_keys=batch_input_keys,
                         collate_fn=collate_fn)
        self.replacing_rate_scheduler = replacing_rate_scheduler

    def _train_update(self, model, optimizer, loss, scheduler):
        '''
        Tranining update
        '''
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        optimizer.step()  # Update weights
        if self.scheduler_on_batch:
            scheduler.step()  # Update learning rate schedule
        self.replacing_rate_scheduler.step()  # Update replace rate scheduler
        model.zero_grad()  # Reset gradients to zero
        self.global_step += 1
        self.records['loss_meter'].update(loss, n=1)
        self.tb_writer.add_scalar('loss', loss, self.global_step)
        self.tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], self.global_step)


def main():
    parser = build_argparse()
    # bert for theseus
    parser.add_argument('--replacing_rate', default=0.3, required=True, type=float,
                        help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    parser.add_argument("--scheduler_type", default='none', choices=['none', 'linear'], help="Scheduler function.")
    parser.add_argument("--scheduler_linear_k", default=0, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--steps_for_replacing", default=0, type=int,
                        help="Steps before entering successor fine_tuning (only useful for constant replacing)")
    parser.add_argument('--predecessor_model_path', type=str, required=True)
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
    processor = Sst2Processor(data_dir=args.data_dir, tokenizer=tokenizer, prefix=prefix)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path, num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_hidden_states = True
    model = model_class.from_pretrained(args.predecessor_model_path, config=config)
    scc_n_layer = model.bert.encoder.scc_n_layer
    model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
    model.to(args.device)
    # trainer
    logger.info("initializing traniner")
    # Replace rate scheduler
    if args.scheduler_type == 'linear':
        replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                              base_replacing_rate=args.replacing_rate,
                                                              k=args.scheduler_linear_k)
    elif args.scheduler_type == 'none':
        replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                replacing_rate=args.replacing_rate,
                                                                replacing_steps=args.steps_for_replacing)
    trainer = TheseusTrainer(logger=logger, args=args,
                             batch_input_keys=processor.get_batch_keys(),
                             replacing_rate_scheduler=replacing_rate_scheduler,
                             collate_fn=processor.collate_fn,
                             metrics=[Accuracy()])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.tsv', 'train')
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.tsv', 'dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
    main()
