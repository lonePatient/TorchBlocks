import os
import sys
import json
import argparse
from pathlib import Path
from .io_utils import check_file, load_json, build_dir, load_yaml
from .import_utils import import_modules_from_file


class Argparser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(Argparser, self).__init__(**kwargs)

    @classmethod
    def build_parser(cls, description='Arguments'):
        parser = cls(description=description, add_help=True)
        parser.arguments_required()
        parser.arguments_common()
        parser.arguments_input_file()
        parser.arguments_dataset()
        parser.arguments_dataloader()
        parser.arguments_pretrained()
        parser.arguments_ema()
        parser.arguments_swa()
        parser.arguments_rdrop()
        parser.arguments_attack()
        parser.arguments_optimimzer_and_scheduler()
        parser.arguments_mixed_precision()
        parser.arguments_model_checkpoint()
        parser.arguments_earlystopping()
        return parser

    @classmethod
    def build_args_from_parser(cls, parser):
        args = parser.parse_args()
        parser.build_experiment_dir(args)
        parser.save_args_to_json(args)
        return args

    @classmethod
    def build_arguments(cls):
        parser = cls.build_parser()
        args = cls.build_args_from_parser(parser)
        return args

    @classmethod
    def build_args_from_file(cls, file_name):
        if isinstance(file_name, Path):
            file_name = str(file_name)
        check_file(file_name)
        fileExtname = os.path.splitext(file_name)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')
        if fileExtname in ['.py']:
            module_nanme, mod = import_modules_from_file(file_name)
            opts_dict = {}
            for name, value in mod.__dict__.items():
                opts_dict[name] = value
            # delete imported module
            del sys.modules[module_nanme]
        elif fileExtname in ['.json']:
            opts_dict = load_json(file_name)
        else:
            opts_dict = load_yaml(file_name)
        return argparse.Namespace(**opts_dict)

    def arguments_required(self):
        group = self.add_argument_group(title="required arguments", description="required arguments")
        group.add_argument('-t', "--task_name", default=None, type=str, required=True,
                           help="The name of the task to train. ")
        group.add_argument('-o', "--output_dir", default=None, type=str, required=True,
                           help="directory to save experiment logs and checkpoints.")
        group.add_argument('-m', "--model_type", default=None, type=str, required=True,
                           help="The name of the model to train.")
        group.add_argument('-d', "--data_dir", default=None, type=str, required=True,
                           help="The input data dir. Should contain the training files for task.")

    def arguments_common(self):
        group = self.add_argument_group(title="common arguments", description="common arguments")
        group.add_argument(
            "--seed", type=int, default=42, help="random seed for initialization")
        group.add_argument(
            "--epoch_seed", action="store_true", help="Whether to seed+ every epoch.")
        group.add_argument(
            "--do_train", action="store_true", help="Whether to run training.")
        group.add_argument(
            "--do_eval", action="store_true", help="Whether to run eval.")
        group.add_argument(
            "--do_predict", action="store_true", help="Whether to run predict.")
        group.add_argument(
            "--device_id", type=str, default='0',
            help='cuda device string. Multi-gpu:"0,1,.." or single-gpu:"0" or cpu:"cpu"')
        group.add_argument(
            '-c', '--config_path', type=str, default=None, help="configuration YAML file")
        group.add_argument(
            '-ss', "--save_steps", type=int, default=-1,
            help="Save checkpoint every X updates steps. ``-1`` means that a epoch")
        group.add_argument(
            '-ls', "--logging_steps", type=int, default=-1,
            help="Log every X updates steps.``-1`` means that a epoch, not used if logging_strategy is `epoch`")
        group.add_argument(
            '-lsy', '--logging_strategy', default=None, type=str, choices=[None, 'epoch'])
        group.add_argument(
            '-exp_name', '--experiment_name', type=str, default='v0', help='experiment name')
        group.add_argument(
            '--log_writer', default='file', choices=['file', 'tensorboard'])
        group.add_argument(
            '--local_rank', type=str, default='0')
        group.add_argument(
            '-f', '--force', default=None, help='overwrite the output directory if it exists.'
        )

    def arguments_input_file(self):
        group = self.add_argument_group(title="input file arguments", description="input file arguments")
        group.add_argument('-train_file', "--train_input_file", default=None, type=str,
                           help="The name of train input file")
        group.add_argument('-eval_file', "--eval_input_file", default=None, type=str,
                           help="The name of eval input file")
        group.add_argument('-test_file', "--test_input_file", default=None, type=str,
                           help="The name of test input file")
        group.add_argument('-label_file', "--label_file_path", default=None, type=str,
                           help="The name of label input file")

    def arguments_dataset(self):
        group = self.add_argument_group(title="dataset arguments", description="dataset arguments")
        group.add_argument('-train_len', "--train_max_seq_length", default=128, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument('-eval_len', "--eval_max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument('-test_len', "--test_max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument('-train_bs', "--per_gpu_train_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for training.")
        group.add_argument('-eval_bs', "--per_gpu_eval_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for evaluation.")
        group.add_argument('-test_bs', "--per_gpu_test_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for test evaluation.")
        group.add_argument("--overwrite_data_cache", action='store_true',
                           help="Whether to overwrite the cached training and evaluation feature sets")
        group.add_argument("--use_data_cache", action='store_true',
                           help='Whether to load the cached training feature sets')
        group.add_argument('--cached_features_file', default=None, type=str, help='custom cached feature file')
        group.add_argument('--max_examples', default=None, type=int, help='debug')
        group.add_argument('--dynamical_padding',default=True,type=bool,
                           help='If dynamical_padding is False, uniform length sequences in batches. Default True')

    def arguments_dataloader(self):
        group = self.add_argument_group(title="dataloader arguments", description="dataloader arguments")
        group.add_argument('--pin_memory', default=False, action='store_true',
                           help='Use pin memory option in data loader')
        group.add_argument('-train_dl', "--train_drop_last", default=False, action='store_true')
        group.add_argument('-eval_dl', "--eval_drop_last", default=False, action='store_true')
        group.add_argument('-test_dl', "--test_drop_last", default=False, action='store_true')
        group.add_argument('--num_workers', default=0, type=int, help='Number of data workers')
        group.add_argument("--persistent_workers", default=False, action="store_true")

    def arguments_pretrained(self):
        group = self.add_argument_group(title="pretrained arguments", description="pretrained arguments")
        group.add_argument("--pretrained_model_path", default=None, type=str,
                           help="Path to pre-trained model selected in the list")
        group.add_argument("--pretrained_config_path", default=None, type=str,
                           help="Pretrained config path if not the same as model_name")
        group.add_argument("--pretrained_tokenizer_path", default=None, type=str,
                           help="Pretrained tokenizer path if not the same as model_name")
        group.add_argument("--do_lower_case", action="store_true",
                           help="Set this flag if you are using an uncased model.")

    def arguments_optimimzer_and_scheduler(self):
        group = self.add_argument_group(title='optimizer and scheduler', description='Optimizer related arguments')
        group.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay for optimizer.")
        group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm. common: 1,10,100,1000")
        group.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for optimizer")
        group.add_argument("--adam_beta2", default=0.999, type=float, help='Beta2 for optimizer')
        group.add_argument("--adam_epsilon", default=1e-8, type=float,
                           help="Epsilon for optimizer. common: 1e-6,1e-7,5e-7,1e-8")
        group.add_argument('--num_cycles', default=0.5, type=float,
                           help='The number of waves in the cosine schedule,common:0.5、1')
        group.add_argument('--min_lr', default=1e-7, type=float, help='Minimum learning rate. common: 1e-7,1e-8')
        group.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate.")
        group.add_argument("--other_learning_rate", default=0.0, type=float, help='other learning rate')
        group.add_argument("--base_model_name", default='base_model', type=str, help='The main body of the model.')
        group.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs")
        group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.", )
        group.add_argument("--warmup_rate", default=0.0, type=float,
                           help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
        group.add_argument("--warmup_steps", default=0, type=int, help='Linear warmup over warmup_steps.')
        group.add_argument("--scheduler_type", default='linear', type=str,
                           choices=["linear", 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                    'constant_with_warmup', 'one_cycle_cosine'],
                           help='The scheduler type to use.')
        group.add_argument('--scheduler_on', default='batch', type=str, choices=['batch', 'epoch'],
                           help='scheduler to start')
        group.add_argument("--scheduler_metric", default=None, type=str)

    def arguments_ema(self):
        group = self.add_argument_group(title='EMA', description='Exponential moving average arguments')
        group.add_argument('--do_ema', action='store_true', help='Exponential moving average')
        group.add_argument('--swa_start', type=int, default=-1, help='EMA start')

    def arguments_swa(self):
        group = self.add_argument_group(title='SWA', description='swa arguments')
        group.add_argument('--do_swa', action='store_true', help='SWA')
        group.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay')

    def arguments_rdrop(self):
        group = self.add_argument_group(title='Rdrop', description='Rdrop arguments')
        group.add_argument("--do_rdrop", action="store_true", help="Whether to run rdrop training.")
        group.add_argument('--rdrop_weight', type=float, default=0.0, help="The weight of rdrop loss")
        group.add_argument('--rdrop_start_epoch', type=int, default=1, help='the epoch to start do rdrop')

    def arguments_attack(self):
        group = self.add_argument_group(title='Adversarial training', description='Adversarial training arguments')
        group.add_argument('--do_fgm', action='store_true', help='Adversarial training')
        group.add_argument('--fgm_name', default='word_embeddings', type=str,
                           help='name for attacks layer,`FGM` use word_embeddings')
        group.add_argument('--fgm_epsilon', default=1.0, type=float, help='attack epsilon,such as 1e-2,1e-3')

        group.add_argument('--do_pgd', action='store_true', help='Adversarial training')
        group.add_argument('--pgd_name', default='word_embeddings', type=str,
                           help='name for attacks layer,`PGD` use word_embeddings')
        group.add_argument('--pgd_epsilon', default=1.0, type=float, help='attack epsilon,such as 1e-2,1e-3')
        group.add_argument('--pgd_number', default=1, type=int, help='the number of attack')
        group.add_argument('--pgd_alpha', default=0.3, type=float, help='attack alpha (lr),such as 1e-4,5e-4,1e-5')

        group.add_argument('--do_awp', action='store_true', help='Adversarial training')
        group.add_argument('--awp_number', default=1, type=int, help='the number of attack')
        group.add_argument('--awp_name', default='weight', type=str, help='name for attacks layer, `AWP` use weight')
        group.add_argument('--awp_epsilon', default=1.0, type=float, help='attack epsilon,such as 1e-2,1e-3')
        group.add_argument('--awp_alpha', default=0.3, type=float, help='attack alpha (lr),such as 1e-4,5e-4,1e-5')
        group.add_argument('--awp_start_step', default=-1, type=int,
                           help='the step to start attack,``-1`` means that no limits')
        group.add_argument('--awp_start_epoch', default=-1, type=int,
                           help='the epoch to start attack,``-1`` means that no limits')
        group.add_argument('--awp_start_score', default=-1, type=float,
                           help='the score to start accack,``-1`` means that no limits')
        group.add_argument('--awp_score_mode', default='min', help='attack score mode')
        group.add_argument('--awp_score_monitor', default='eval_loss', help='attack score monitor')

    def arguments_mixed_precision(self):
        group = self.add_argument_group(title="mixed precision arguments", description="mixed precision arguments")
        group.add_argument("--do_fp16", action="store_true",
                           help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
        group.add_argument("--fp16_opt_level", type=str, default="O1",
                           help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html", )
        group.add_argument('--fp16_backend', default='apex', type=str, choices=['apex', 'amp', 'auto'],
                           help="The backend to be used for mixed precision.")
        group.add_argument("--fp16_full_eval", action='store_true',
                           help="Whether to use full 16-bit precision evaluation instead of 32-bit")

    def arguments_model_checkpoint(self):
        group = self.add_argument_group(title='model checkpoint', description='model checkpoint arguments')
        group.add_argument('-ckpt_d', "--checkpoint_mode", default='min', type=str, help='model checkpoint mode')
        group.add_argument('-ckpt_m', "--checkpoint_monitor", default='eval_loss', type=str,
                           help='model checkpoint monitor')
        group.add_argument('-ckpt_s', "--checkpoint_save_best", action='store_true', help='Whether to save best model')
        group.add_argument('-ckpt_v', "--checkpoint_verbose", default=1, type=int,
                           help='whether to print checkpoint info')
        group.add_argument("-ckpt_p", "--checkpoint_predict_code", type=str, default=None,
                           help='The version of checkpoint to predict')
        group.add_argument('--eval_all_checkpoints', action="store_true", help="Evaluate all checkpoints starting", )

    def arguments_earlystopping(self):
        group = self.add_argument_group(title='early stopping', description='early stopping arguments')
        group.add_argument("--earlystopping_patience", default=-1, type=int,
                           help='Interval (number of epochs) between checkpoints,``-1`` means that no earlystopping')
        group.add_argument("--earlystopping_mode", default='min', type=str, help='early stopping mode')
        group.add_argument("--earlystopping_monitor", default='eval_loss', type=str, help='early stopping monitor')
        group.add_argument("--earlystopping_verbose", default=1, type=int, help='whether to print earlystopping info')
        group.add_argument('--earlystopping_save_state_path', default=None, type=str)
        group.add_argument('--earlystopping_load_state_path', default=None, type=str)

    def save_args_to_json(self, args):
        if args.do_train:
            save_arguments_file_name = f"{args.task_name}_{args.model_type}_{args.experiment_name}_options.json"
            save_arguments_file_path = os.path.join(args.output_dir, save_arguments_file_name)
            if os.path.exists(save_arguments_file_path):
                print(f"[Warning]File {save_arguments_file_path} exist,Overwrite arguments file")
            with open(str(save_arguments_file_path), 'w') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)

    def print_args(self, args):
        print('**********************************')
        print('************ Arguments ***********')
        print('**********************************')
        args_list = sorted(args.__dict__.items(), key=lambda x: x[0])
        msg = ''
        for k, v in args_list:
            msg += f' - {k}: {v}\n'
        print(msg)

    def build_experiment_dir(self, args):
        _name = f'{args.task_name}_{args.model_type}_{args.experiment_name}'
        args.output_dir = os.path.join(args.output_dir, _name)
        build_dir(args.output_dir, exist_ok=True)

