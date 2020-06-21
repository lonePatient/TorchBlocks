import os
import csv
import torch
from torchblocks.metrics import MattewsCorrcoef
from torchblocks.trainer import TextClassifierTrainer
from torchblocks.callback import ModelCheckpoint, TrainLogger
from torchblocks.processor import TextClassifierProcessor, InputExample
from torchblocks.utils import seed_everything, dict_to_text, build_argparse
from torchblocks.utils import prepare_device, get_checkpoints
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import WEIGHTS_NAME

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


class ColaProcessor(TextClassifierProcessor):
    def __init__(self, tokenizer, data_dir, logger, prefix):
        super().__init__(tokenizer=tokenizer, data_dir=data_dir, logger=logger, prefix=prefix)

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
            label = line[1] if set_type != 'test' else None
            examples.append(
                InputExample(guid=guid, texts=[text_a,None], label=label))
        return examples


class FreelbTrainer(TextClassifierTrainer):
    def __init__(self, args, metrics, logger, batch_input_keys,collate_fn=None):
        super().__init__(args=args,
                         metrics=metrics,
                         logger=logger,
                         batch_input_keys=batch_input_keys,
                         collate_fn=collate_fn)

    def _train_step(self, model, batch, optimizer):
        model.train()
        inputs = self.build_inputs(batch)
        input_ids = inputs['input_ids']
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = model.module.bert.embeddings.word_embeddings(input_ids)
        else:
            embeds_init = model.bert.embeddings.word_embeddings(input_ids)
        if self.args.adv_init_mag > 0:
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.args.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.args.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.args.adv_init_mag,
                                                               self.args.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)
        for astep in range(self.args.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.args.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.args.adv_max_norm).to(embeds_init)
                    reweights = (self.args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.args.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.args.adv_max_norm, self.args.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.args.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = model.module.bert.embeddings.word_embeddings(input_ids)
            else:
                embeds_init = model.bert.embeddings.word_embeddings(input_ids)
        return loss.item()


def main():
    parser = build_argparse()
    parser.add_argument('--adv_lr', type=float, default=1e-2)
    parser.add_argument('--adv_K', type=int, default=3, help="should be at least 1")
    parser.add_argument('--adv_init_mag', type=float, default=2e-2)
    parser.add_argument('--adv_norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)
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
    processor = ColaProcessor(tokenizer, args.data_dir, logger, prefix=prefix)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    logger.info("initializing model and config")
    config = config_class.from_pretrained(args.model_path,
                                          num_labels=num_labels,
                                          attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                                          hidden_dropout_prob=args.hidden_dropout_prob,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(args.device)

    logger.info("initializing traniner")
    trainer = FreelbTrainer(logger=logger,
                            args=args,
                            batch_input_keys=processor.get_batch_keys(),
                            collate_fn=processor.collate_fn,
                            metrics=[MattewsCorrcoef()])
    if args.do_train:
        train_dataset = processor.create_dataset(max_seq_length=args.train_max_seq_length,
                                                 data_name='train.tsv', mode='train')
        eval_dataset = processor.create_dataset(max_seq_length=args.eval_max_seq_length,
                                                data_name='dev.tsv', mode='dev')
        trainer.train(model, train_dataset=train_dataset, eval_dataset=eval_dataset)

    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        eval_dataset = processor.create_dataset(max_seq_length=args.eval_max_seq_length,
                                                data_name='dev.tsv', mode='dev')
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints or args.checkpoint_number > 0:
            checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
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
                                                data_name='test.tsv', mode='test')
        if args.checkpoint_number == 0:
            raise ValueError("checkpoint number should > 0,but get %d", args.checkpoint_number)
        checkpoints = get_checkpoints(args.output_dir, args.checkpoint_number, WEIGHTS_NAME)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            trainer.predict(model, test_dataset=test_dataset, prefix=str(global_step))


if __name__ == "__main__":
    main()
