import os
import csv
import torch
import copy
from torch.nn import MSELoss
from torchblocks.metrics import MattewsCorrcoef
from torchblocks.trainer import TextClassifierTrainer
from torchblocks.callback import TrainLogger
from torchblocks.processor import TextClassifierProcessor, InputExample
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, WEIGHTS_NAME

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}
'''
Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation
'''


class ColaProcessor(TextClassifierProcessor):
    def __init__(self, tokenizer, data_dir, prefix):
        super().__init__(tokenizer=tokenizer, data_dir=data_dir, prefix=prefix)

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[3] if set_type != 'test' else line[1]
            label = line[1] if set_type != 'test' else "0"
            examples.append(
                InputExample(guid=guid, texts=[text_a, None], label=label))
        return examples


class SDATrainer(TextClassifierTrainer):
    def __init__(self, args, metrics, logger, kd_model, kd_loss_fct, batch_input_keys, collate_fn=None):
        super().__init__(args=args, metrics=metrics, logger=logger,
                         batch_input_keys=batch_input_keys,
                         collate_fn=collate_fn)
        self.kd_model = kd_model
        self.kd_loss_fct = kd_loss_fct

    def _train_step(self, model, batch, optimizer):
        model.train()
        inputs = self.build_inputs(batch)
        outputs = model(**inputs)
        loss, logits = outputs[:2]
        self.kd_model.eval()
        with torch.no_grad():
            outputs = self.kd_model(**inputs)
            kd_logits = outputs[1]
        kd_loss = self.kd_loss_fct(logits, kd_logits)
        loss += self.args.kd_coeff * kd_loss
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.item()

    def _train_update(self, model, optimizer, loss, scheduler):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        optimizer.step()
        if self.scheduler_on_batch:
            scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        self.global_step += 1
        self.records['loss_meter'].update(loss, n=1)
        self.tb_writer.add_scalar('loss', loss, self.global_step)
        self.tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], self.global_step)
        # update kd_model parameters
        one_minus_decay = 1.0 - self.args.kd_decay
        self.kd_model.eval()
        with torch.no_grad():
            parameters = [p for p in model.parameters() if p.requires_grad]
            for s_param, param in zip(self.kd_model.parameters(), parameters):
                s_param.sub_(one_minus_decay * (s_param - param))


def main():
    parser = build_argparse()
    parser.add_argument('--kd_decay', type=float, default=0.999)
    parser.add_argument('--kd_coeff', type=float, default=1.0)
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
    processor = ColaProcessor(tokenizer, args.data_dir, prefix=prefix)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path, num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    # trainer
    logger.info("initializing traniner")
    trainer = SDATrainer(logger=logger, args=args, collate_fn=processor.collate_fn,
                         batch_input_keys=processor.get_batch_keys(),
                         kd_model=copy.deepcopy(model),
                         kd_loss_fct=MSELoss(),
                         metrics=[MattewsCorrcoef()])
    # do train
    if args.do_train:
        train_dataset = processor.create_dataset(args.train_max_seq_length, 'train.tsv', 'train')
        eval_dataset = processor.create_dataset(args.eval_max_seq_length, 'dev.tsv', 'dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)
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
