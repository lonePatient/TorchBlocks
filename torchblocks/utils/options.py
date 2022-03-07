import os
import json
import logging
import argparse
from pathlib import Path
from torchblocks.utils.paths import save_json, check_file

logger = logging.getLogger(__name__)


class Argparser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(Argparser, self).__init__(**kwargs)

    @classmethod
    def get_training_parser(cls, description='Arguments'):
        parser = cls(description=description, add_help=True)
        parser.arguments_required()
        parser.arguments_common()
        parser.arguments_input_file()
        parser.arguments_dataset()
        parser.arguments_dataloader()
        parser.arguments_pretrained()
        parser.arguments_ema()
        parser.arguments_adv()
        parser.arguments_optimimzer()
        parser.arguments_lr_scheduler()
        parser.arguments_apex()
        parser.arguments_checkpoint()
        parser.arguments_earlystopping()
        return parser

    @classmethod
    def parse_args_from_parser(cls, parser):
        args = parser.parse_args()
        parser.make_experiment_dir(args)
        parser.save_args_to_json(args)
        parser.print_args(args)
        return args

    @classmethod
    def parse_args_from_json(cls, json_file):
        check_file(json_file)
        data = json.loads(Path(json_file).read_text())
        return argparse.Namespace(**data)

    @classmethod
    def get_training_arguments(cls):
        parser = cls.get_training_parser()
        args = cls.parse_args_from_parser(parser)
        return args

    def get_val_argments(self):
        args = Argparser.get_training_arguments()
        return args

    def get_predict_arguments(self):
        args = Argparser.get_training_arguments()
        return args

    def arguments_required(self):
        group = self.add_argument_group(title="required arguments", description="required arguments")
        group.add_argument("--task_name", default=None, type=str, required=True,
                           help="The name of the task to train. ")
        group.add_argument("--output_dir", default=None, type=str, required=True,
                           help="The output directory where the model predictions and checkpoints will be written.")
        group.add_argument("--model_type", default=None, type=str, required=True,
                           help="The name of the model to train.")
        group.add_argument("--data_dir", default=None, type=str, required=True,
                           help="The input data dir. Should contain the training files for task.")

    def arguments_common(self):
        group = self.add_argument_group(title="common arguments", description="common arguments")
        group.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        group.add_argument("--do_train", action="store_true", help="Whether to run training.")
        group.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        group.add_argument("--do_predict", action="store_true", help="Whether to run predict on the test set.")
        group.add_argument("--device_id", type=str, default='0',
                           help='multi-gpu:"0,1,.." or single-gpu:"0" or cpu:"cpu"')
        group.add_argument("--evaluate_during_training", action="store_true",
                           help="Whether to run evaluation during training at each logging step.", )
        group.add_argument('--load_arguments_file', type=str, default=None, help="load args from arguments file")
        group.add_argument("--save_steps", type=int, default=-1,
                           help="Save checkpoint every X updates steps. ``-1`` means that a epoch")
        group.add_argument("--logging_steps", type=int, default=-1,
                           help="Log every X updates steps.``-1`` means that a epoch")
        group.add_argument('--experiment_code', type=str, default='v0', help='experiment code')

    def arguments_input_file(self):
        group = self.add_argument_group(title="input file arguments", description="input file arguments")
        group.add_argument("--train_input_file", default=None, type=str, help="The name of train input file")
        group.add_argument("--eval_input_file", default=None, type=str, help="The name of eval input file")
        group.add_argument("--test_input_file", default=None, type=str, help="The name of test input file")

    def arguments_dataset(self):
        group = self.add_argument_group(title="datasets arguments", description="datasets arguments")
        group.add_argument("--train_max_seq_length", default=128, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument("--eval_max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument("--test_max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for training.")
        group.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for evaluation.")
        group.add_argument("--per_gpu_test_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for test evaluation.")
        group.add_argument("--overwrite_data_cache", action='store_true',
                           help="Whether to overwrite the cached training and evaluation feature sets")
        group.add_argument("--use_data_cache", action='store_true',
                           help='Whether to load the cached training feature sets')

    def arguments_dataloader(self):
        group = self.add_argument_group(title="dataloader arguments", description="dataloader arguments")
        group.add_argument('--pin_memory', default=False, action='store_true',
                           help='Use pin memory option in data loader')
        group.add_argument("--drop_last", default=False, action='store_true')
        group.add_argument('--num_workers', default=0, type=int, help='Number of data workers')
        group.add_argument("--persistent_workers", default=False, action="store_true", help="")

    def arguments_pretrained(self):
        group = self.add_argument_group(title="pretrained arguments", description="pretrained arguments")
        group.add_argument("--pretrained_model_path", default=None, type=str,
                           help="Path to pre-trained model or shortcut name selected in the list")
        group.add_argument("--pretrained_config_name", default=None, type=str,
                           help="Pretrained config name or path if not the same as model_name")
        group.add_argument("--pretrained_tokenizer_name", default=None, type=str,
                           help="Pretrained tokenizer name or path if not the same as model_name")
        group.add_argument("--do_lower_case", action="store_true",
                           help="Set this flag if you are using an uncased model.")
        group.add_argument("--pretrained_cache_dir", default=None, type=str,
                           help="Where do you want to store the pre-trained models downloaded from s3", )

    def arguments_ema(self):
        group = self.add_argument_group(title='EMA', description='Exponential moving average arguments')
        group.add_argument('--ema_enable', action='store_true', help='Exponential moving average')
        group.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay')
        group.add_argument("--model_ema_force_cpu", action='store_true')

    def arguments_adv(self):
        group = self.add_argument_group(title='Adversarial training', description='Adversarial training arguments')
        group.add_argument('--adv_enable', action='store_true', help='Adversarial training')
        group.add_argument('--adv_type', default='fgm', type=str, choices=['fgm', 'pgd'])
        group.add_argument('--adv_epsilon', type=float, default=1.0, help='adv epsilon')
        group.add_argument('--adv_name', type=str, default='word_embeddings',
                           help='name for adversarial layer')
        group.add_argument('--adv_number', default=1, type=int, help='the number of attack')
        group.add_argument('--adv_alpha', default=0.3, type=float, help='adv alpha')

    def arguments_optimimzer(self):
        group = self.add_argument_group(title='optimizer', description='Optimizer related arguments')
        group.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
        group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        group.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for AdamW optimizer")
        group.add_argument("--adam_beta2", default=0.999, type=float, help='Beta2 for AdamW optimizer')
        group.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    def arguments_lr_scheduler(self):
        group = self.add_argument_group(title="lr scheduler arguments", description="LR scheduler arguments")
        group.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        group.add_argument("--other_learning_rate", default=0.0, type=float)
        group.add_argument("--base_model_name", default='base_model', type=str, help='The main body of the model.')
        group.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs")
        group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.", )
        group.add_argument("--warmup_proportion", default=0.1, type=float,
                           help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
        group.add_argument("--warmup_steps", default=0, type=int,
                           help='Linear warmup over warmup_steps.')
        group.add_argument("--scheduler_type", default='linear', type=str,
                           choices=["linear", 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                    'constant_with_warmup'],
                           help='The scheduler type to use.')

    def arguments_apex(self):
        group = self.add_argument_group(title="apex arguments", description="apex arguments")
        group.add_argument("--fp16", action="store_true",
                           help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
        group.add_argument("--fp16_opt_level", type=str, default="O1",
                           help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html", )
        group.add_argument('--fp16_backend', default='auto', type=str, choices=["auto", "amp", "apex"],
                           help="The backend to be used for mixed precision.")
        group.add_argument("--fp16_full_eval", action='store_true',
                           help="Whether to use full 16-bit precision evaluation instead of 32-bit")

    def arguments_checkpoint(self):
        group = self.add_argument_group(title='model checkpoint', description='model checkpoint arguments')
        group.add_argument("--checkpoint_mode", default='min', type=str, help='model checkpoint mode')
        group.add_argument("--checkpoint_monitor", default='eval_loss', type=str, help='model checkpoint monitor')
        group.add_argument("--checkpoint_save_best", action='store_true', help='Whether to save best model')
        group.add_argument("--checkpoint_verbose", default=1, type=int, help='whether to print checkpoint info')
        group.add_argument("--checkpoint_predict_code", type=str, default=None,
                           help='The version of checkpoint to predict')
        group.add_argument('--eval_all_checkpoints', action="store_true", help="Evaluate all checkpoints starting", )

    def arguments_earlystopping(self):
        group = self.add_argument_group(title='early stopping', description='early stopping arguments')
        group.add_argument("--earlystopping_patience", default=-1, type=int,
                           help='Interval (number of epochs) between checkpoints')
        group.add_argument("--earlystopping_mode", default='min', type=str, help='early stopping mode')
        group.add_argument("--earlystopping_monitor", default='eval_loss', type=str, help='early stopping monitor')
        group.add_argument("--earlystopping_verbose", default=1, type=int, help='whether to print earlystopping info')
        group.add_argument('--earlystopping_save_state_path', default=None, type=str)
        group.add_argument('--earlystopping_load_state_path', default=None, type=str)

    def save_args_to_json(self, args):
        if args.do_train:
            save_arguments_file_name = f"{args.task_name}_{args.model_type}_{args.experiment_code}_opts.json"
            save_arguments_file_path = os.path.join(args.output_dir, save_arguments_file_name)
            if os.path.exists(save_arguments_file_path):
                logger.info(f"File {save_arguments_file_path} exist,Overwrite arguments file")
            # save_json(vars(args), save_arguments_file_path)
            with open(str(save_arguments_file_path), 'w') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)

    def print_args(self, args):
        print('**********************************')
        print('************ Arguments ***********')
        print('**********************************')
        argskeys = list(args.__dict__.keys())
        argskeys.sort()
        for key in argskeys:
            print('  {}: {}'.format(key, args.__dict__[key]))

    def make_experiment_dir(self, args):
        args.output_dir = os.path.join(args.output_dir, f'{args.task_name}_{args.model_type}_{args.experiment_code}')
        os.makedirs(args.output_dir, exist_ok=True)
