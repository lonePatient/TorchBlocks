import os
import gc
import math
import copy
import warnings
import torch
import torch.nn as nn
from argparse import Namespace
from packaging import version
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from ..losses.kl_divergence import BKL
from ..callback import ModelCheckpoint, EarlyStopping, ProgressBar, EMA, SWA
from ..callback.model_checkpoint import (WEIGHTS_NAME,
                                         TRAINER_STATE_NAME,
                                         OPTIMIZER_NAME,
                                         SCHEDULER_NAME,
                                         SCALER_NAME)
from ..callback.attacks import FGM, PGD, AWP
from ..callback.file_writer import FileWriter
from ..optims.adamw import AdamW
from ..optims.lr_scheduler import get_lr_scheduler
from ..utils.common_utils import (check_object_type,
                                  convert_to_list,
                                  has_key,
                                  check_object_keys)
from ..utils.logger import Logger
from ..utils.meter import AverageMeter
from ..utils.import_utils import is_apex_available
from ..utils.io_utils import (to_json_string,
                              save_pickle,
                              json_to_text,
                              save_json,
                              is_file)
from ..utils.ckpt_utils import load_model
from ..utils.seed import seed_everything
from ..utils.tensor_utils import convert_tensor_list_to_dict, convert_cuda_to_cpu

warnings.filterwarnings('ignore')
if version.parse(torch.__version__) >= version.parse("1.10"):
    torch.set_warn_always(False)

_is_native_amp_available = False
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast, GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


class TrainBaseBuilder:
    """Base class for iterative trainer."""
    # Variable names that are not converted to GPU in batch data，For example 'input_length’
    keys_to_ignore_on_gpu = []
    # Variables that are not stored in the eval and predict process
    keys_to_ignore_on_save_result = ['input_ids', 'token_type_ids']
    # Variables that are not stored in the checkpoint. For example 'optimizer'
    keys_to_ignore_on_save_checkpoint = []
    mode_dict = {'min': torch.lt, 'max': torch.gt}

    def __init__(self,
                 opts,
                 model,
                 metrics,
                 logger,
                 **kwargs):
        '''
        Training master function
        Args:
            opts: options
            model
            metrics
            logger
            **kwargs:
        '''
        self.opts = opts
        self.model = model
        self.logger = logger
        self.metrics = metrics
        self.global_step = 0
        self._init_ema()
        self._init_swa()
        self._init_attack()
        self._init_optimizer()
        self._init_early_stopping()
        self._init_model_checkpoint()
        self.metrics = convert_to_list(self.metrics)
        self.device_num = getattr(opts, 'device_num', 0)
        self.device = getattr(opts, 'device', torch.device("cpu"))
        self.prefix = "_".join([opts.task_name, opts.model_type, opts.experiment_name])
        self.build_log_writer()
        self.build_mixed_precision()
        check_object_type(object=self.model, check_type=nn.Module, name='model')
        check_object_type(object=self.opts, check_type=Namespace, name='self.opts')
        check_object_type(object=self.logger, check_type=Logger, name='self.logger')
        check_object_type(object=self.metrics, check_type=list, name='metric')
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _init_ema(self):
        # EMA
        if self.opts.do_ema:
            self.logger.info('Using EMA training.....')
            self.model_ema = EMA(model=self.model.module if hasattr(self.model, 'module') else self.model,
                                 decay=self.opts.ema_decay)
            self.model_ema.register()

    def _init_swa(self):
        # SWA
        if self.opts.do_swa:
            self.logger.info('Using SWA training.....')
            self.model_swa = copy.deepcopy(self.model)

    def _init_attack(self):
        # Adversarial training
        msg = f"Adversarial training. FGM: {self.opts.do_fgm} PGD: {self.opts.do_pgd} AWP: {self.opts.do_awp}"
        self.logger.info(msg)
        self.attack_models = self.build_attack_model()

    def _init_optimizer(self):
        # optimizer
        self.optimizer = self.build_optimizer(self.model)

    def _init_model_checkpoint(self):
        # checkpoint
        self.model_checkpoint = ModelCheckpoint(
            logger=self.logger,
            mode=self.opts.checkpoint_mode,
            ckpt_dir=self.opts.output_dir,
            monitor=self.opts.checkpoint_monitor,
            verbose=self.opts.checkpoint_verbose,
            save_best=self.opts.checkpoint_save_best,
            keys_to_ignore_on_save=self.keys_to_ignore_on_save_checkpoint
        )

    def _init_early_stopping(self):
        # earlystopping
        self.early_stopping = None
        if self.opts.earlystopping_patience > 0:
            msg = f"`EarlyStopping patience` is {self.opts.earlystopping_patience},using early stopping."
            self.logger.info(msg)
            self.early_stopping = EarlyStopping(
                mode=self.opts.earlystopping_mode,
                patience=self.opts.earlystopping_patience,
                monitor=self.opts.earlystopping_monitor,
                save_state_path=self.opts.earlystopping_save_state_path,
                load_state_path=self.opts.earlystopping_load_state_path
            )

    def build_mixed_precision(self):
        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None
        if self.opts.do_fp16:
            if self.opts.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = self.opts.fp16_backend
            self.logger.info(f"Using {self.fp16_backend} fp16 backend")
            if self.fp16_backend == "amp":
                self.use_amp = True
                self.scaler = GradScaler()
            else:
                if not is_apex_available():
                    msg = ("Using FP16 with APEX but APEX is not installed, "
                           "please refer to https://www.github.com/nvidia/apex.")
                    raise ImportError(msg)
                self.use_apex = True

    # TODO:If there are multiple Adversarial learning methods, consider the order of methods
    def build_attack_model(self):
        attack_models = {}
        if self.opts.do_fgm:
            attack_model = FGM(self.model, self.opts.fgm_name, self.opts.fgm_epsilon)
            attack_models['fgm'] = attack_model
        if self.opts.do_pgd:
            attack_model = PGD(self.model, self.opts.pgd_name, self.opts.pgd_epsilon, self.opts.pgd_alpha)
            attack_models['pgd'] = attack_model
        if self.opts.do_awp:
            attack_model = AWP(self.model, self.opts.awp_name, self.opts.awp_epsilon, self.opts.awp_alpha,
                               self.opts.awp_start_epoch, self.opts.awp_start_step, self.opts.awp_start_score,
                               self.opts.awp_score_mode)
            attack_models['awp'] = attack_model
        return attack_models

    def build_record_tracker(self, **kwargs):
        self.record_tracker = {}
        self.record_tracker['result'] = {}
        for key, value in kwargs.items():
            if key not in self.record_tracker:
                self.record_tracker[key] = value

    def build_record_meter(self, key, value=None, n=1):
        if key not in self.record_tracker:
            self.record_tracker[key] = AverageMeter()
            if value is not None:
                self.record_tracker[key].update(value, n=n)
        else:
            self.record_tracker[key].update(value, n=n)

    def build_log_writer(self):
        # tensorboard
        if _has_tensorboard and self.opts.log_writer == 'tensorboard':
            msg = f'Initializing summary writer for tensorboard with log_dir={self.opts.output_dir}'
            self.logger.info(msg)
            exp_dir = os.path.join(self.opts.output_dir, f'{self.prefix}_tb_logs')
            self.writer = SummaryWriter(log_dir=exp_dir, comment='Training logs')
            self.writer.add_text("train_arguments", to_json_string(self.opts.__dict__))
        elif self.opts.log_writer == 'file':
            exp_dir = os.path.join(self.opts.output_dir, f'{self.prefix}_file_logs')
            self.writer = FileWriter(log_dir=exp_dir)
        else:
            # TODO: Add WB
            pass

    def reset_metrics(self):
        '''
        The `metric` class must contain the `reset` function
        Returns:
        '''
        for metric in self.metrics:
            if not hasattr(metric, 'reset'):
                msg = "module 'metric' has no attribute 'reset'"
                return ValueError(msg)
            metric.reset()

    def _param_optimizer(self, params, learning_rate, no_decay, weight_decay):
        _params = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay,
             'lr': learning_rate},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': learning_rate},
        ]
        return _params

    def build_model_param_optimizer(self, model):
        '''
       If you need to assign different learning rates to different models,
       In the `transformer` module,specify `base_model_name`, the default is `base_model_name=`base_model`.
       For base_model use learning_rate, for the rest use other_learning_rate
        '''
        no_decay = ["bias", 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        if (
                hasattr(model, self.opts.base_model_name)
                and self.opts.other_learning_rate > 0.0
        ):
            msg = (
                f"The initial learning rate for {self.opts.base_model_name} model params : {self.opts.learning_rate} ,"
                f"and other model params is : {self.opts.other_learning_rate}"
            )
            self.logger.info(msg)
            base_model = getattr(model, self.opts.base_model_name)
            base_model_param = list(base_model.named_parameters())
            base_model_param_ids = [id(p) for n, p in base_model_param]
            other_model_param = [(n, p) for n, p in model.named_parameters() if
                                 id(p) not in base_model_param_ids]
            optimizer_grouped_parameters.extend(
                self._param_optimizer(base_model_param, self.opts.learning_rate, no_decay,
                                      self.opts.weight_decay))
            optimizer_grouped_parameters.extend(
                self._param_optimizer(other_model_param, self.opts.other_learning_rate, no_decay,
                                      self.opts.weight_decay))
        else:
            all_model_param = list(model.named_parameters())
            optimizer_grouped_parameters.extend(
                self._param_optimizer(all_model_param, self.opts.learning_rate, no_decay, self.opts.weight_decay))
        return optimizer_grouped_parameters

    def build_optimizer(self, model):
        '''
        Setup the optimizer.
        '''
        self.logger.info("The custom optimizer is None, using default `AdamW` optimizer")
        optimizer_grouped_parameters = self.build_model_param_optimizer(model)
        optimizer = AdamW(params=optimizer_grouped_parameters,
                          lr=self.opts.learning_rate,
                          eps=self.opts.adam_epsilon,
                          betas=(self.opts.adam_beta1, self.opts.adam_beta2),
                          weight_decay=self.opts.weight_decay)
        return optimizer

    def build_warmup_steps(self):
        """
        Get number of steps used for a linear warmup.
        """
        if self.opts.warmup_rate < 0 or self.opts.warmup_rate > 1:
            raise ValueError("warmup_rate must lie in range [0,1]")
        elif self.opts.warmup_rate > 0 and self.opts.warmup_steps > 0:
            msg = ("Both warmup_rate and warmup_steps given, "
                   "warmup_steps will override any effect of warmup_rate during training")
            self.logger.info(msg)
        warmup_steps = (
            self.opts.warmup_steps if self.opts.warmup_steps > 0 else math.ceil(
                self.num_update_training_steps * self.opts.warmup_rate)
        )
        return warmup_steps

    def build_lr_scheduler(self):
        '''
        the learning rate scheduler.
        '''
        scheduler_function = get_lr_scheduler(self.opts.scheduler_type)
        warmup_steps = self.build_warmup_steps()
        scheduler = scheduler_function(optimizer=self.optimizer,
                                       num_warmup_steps=warmup_steps,
                                       num_training_steps=self.num_update_training_steps,
                                       num_cycles=self.opts.num_cycles)
        return scheduler

    def build_train_dataloader(self, train_data):
        '''
        Load train dataset
        '''
        if isinstance(train_data, DataLoader):
            data_loader = train_data
        elif isinstance(train_data, Dataset):
            batch_size = self.opts.per_gpu_train_batch_size * max(1, self.device_num)
            sampler = RandomSampler(train_data)
            if hasattr(train_data, 'build_train_sampler'):
                sampler = train_data.build_train_sampler
            collate_fn = train_data.build_data_collator
            if hasattr(train_data, "build_train_collator"):
                collate_fn = train_data.build_train_collator
            data_loader = DataLoader(train_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     pin_memory=self.opts.pin_memory,
                                     drop_last=self.opts.train_drop_last,
                                     num_workers=self.opts.num_workers)
        else:
            raise TypeError("train_data type `{}` not support".format(type(train_data)))
        return data_loader

    def build_eval_dataloader(self, dev_data):
        '''
        Load eval dataset
        '''
        if isinstance(dev_data, DataLoader):
            data_loader = dev_data
        elif isinstance(dev_data, Dataset):
            batch_size = self.opts.per_gpu_eval_batch_size * max(1, self.device_num)
            sampler = SequentialSampler(dev_data)
            if hasattr(dev_data, 'build_eval_sampler'):
                sampler = dev_data.build_eval_sampler
            collate_fn = dev_data.build_data_collator
            if hasattr(dev_data, "build_eval_collator"):
                collate_fn = dev_data.build_eval_collator
            data_loader = DataLoader(dev_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     drop_last=self.opts.eval_drop_last,
                                     pin_memory=self.opts.pin_memory,
                                     num_workers=self.opts.num_workers)
        else:
            raise TypeError("dev_data type `{}` not support".format(type(dev_data)))
        return data_loader

    def build_test_dataloader(self, test_data):
        '''
        Load test dataset
        '''
        if isinstance(test_data, DataLoader):
            data_loader = test_data
        elif isinstance(test_data, Dataset):
            batch_size = self.opts.per_gpu_test_batch_size * max(1, self.device_num)
            sampler = SequentialSampler(test_data)
            if hasattr(test_data, 'build_test_sampler'):
                sampler = test_data.build_test_sampler
            collate_fn = test_data.build_data_collator
            if hasattr(test_data, "build_test_collator"):
                collate_fn = test_data.build_test_collator
            data_loader = DataLoader(test_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     drop_last=self.opts.test_drop_last,
                                     pin_memory=self.opts.pin_memory,
                                     num_workers=self.opts.num_workers)
        else:
            raise TypeError("test_data type `{}` not support".format(type(test_data)))
        return data_loader

    def build_batch_inputs(self, batch):
        '''
        Sent all model inputs to the appropriate device (GPU on CPU)
        rreturn:
         The inputs are in a dictionary format
         keys_to_ignore_on_gpu: Variables stored in the cpu
        '''
        outputs = {}
        for key, value in batch.items():
            if (key not in self.keys_to_ignore_on_gpu) and value is not None:
                outputs[key] = value.to(self.device)
            else:
                outputs[key] = value
        return outputs

    def process_batch_inputs(self, batch):
        '''
        dynamic processing of batches
        Args:
            batch:
        Returns:
        '''
        return batch

    def build_eval_and_save_steps(self):
        if self.opts.logging_strategy == 'epoch' or self.opts.logging_steps <= 0:
            self.opts.logging_steps = self.num_update_steps_per_epoch
        if self.opts.save_steps <= 0:
            self.opts.save_steps = self.num_update_steps_per_epoch

    def build_model_warp(self):
        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opts.fp16_opt_level)
        # Multi-gpu training (should be after apex fp16 initialization)
        if self.device_num > 1:
            self.model = nn.DataParallel(self.model)

    def name_to_metric(self, metric_name):
        check_object_keys(object=self.record_tracker['result'], key=metric_name, msg='Metric Result')
        return self.record_tracker['result'][metric_name]

    def running_scheduler_on_batch(self):
        # Update learning rate schedule
        if self.scheduler:
            if self.opts.scheduler_on == 'batch':
                if self.opts.scheduler_metric is None:
                    self.scheduler.step()
                else:
                    step_metric = self.name_to_metric(self.opts.scheduler_metric)
                    self.scheduler.step(step_metric)

    def running_scheduler_on_epoch(self):
        # Update learning rate schedule
        if self.scheduler is not None:
            if self.opts.scheduler_on == 'epoch':
                if self.opts.scheduler_metric is None:
                    self.scheduler.step()
                else:
                    step_metric = self.name_to_metric(self.opts.scheduler_metric)
                    self.scheduler.step(step_metric)

    def build_state_object(self, **kwargs):
        '''
        save state object
        '''
        states = {
            'opts': self.opts,
            'optimizer': self.optimizer,
            'global_step': self.global_step,
            'model': self.model.module if hasattr(self.model, "module") else self.model
        }
        if self.scheduler is not None:
            states['scheduler'] = self.scheduler
        if self.use_amp:
            states['scaler'] = self.scaler
        for key, value in kwargs.items():
            if key not in states:
                states[key] = value
        return states

    def resume_from_checkpoint(self, resume_path=None):
        '''
        Check if continuing training from a checkpoint
        '''
        if resume_path is not None:
            optimizer_path = os.path.join(resume_path, OPTIMIZER_NAME)
            scheduler_path = os.path.join(resume_path, SCHEDULER_NAME)
            state_path = os.path.join(resume_path, TRAINER_STATE_NAME)
            model_path = os.path.join(resume_path, WEIGHTS_NAME)
            scaler_path = os.path.join(resume_path, SCALER_NAME)
            if is_file(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            if is_file(scheduler_path):
                self.scheduler.load_state_dict(torch.load(scheduler_path))
            if is_file(state_path):
                state = torch.load(state_path)
                if self.model_checkpoint and hasattr(state, 'best_score'):
                    self.model_checkpoint.best = state['best_score']
                del state
                gc.collect()
            if is_file(model_path):
                if self.use_amp and is_file(scaler_path):
                    self.scaler.load_state_dict(torch.load(scaler_path))
                load_model(self.model, model_path, device=self.device)

    def train_rdrop_forward(self, inputs, epoch):
        # rdrop training forward
        rdrop_fct = BKL()
        outputs = self.train_common_forward(inputs)
        if epoch >= self.opts.rdrop_start_epoch:
            outputs_2 = self.train_common_forward(inputs)
            rdrop_loss = rdrop_fct(outputs['logits'], outputs_2['logits'])
            loss_weight = (1 - self.opts.rdrop_weight) / 2
            loss = loss_weight * outputs['loss'] + loss_weight * outputs_2['loss'] + self.opts.rdrop_weight * rdrop_loss
            outputs['loss'] = loss
        return outputs

    def train_common_forward(self, inputs):
        '''
        common training forward
        '''
        self.model.train()
        if self.use_amp:
            with autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        check_object_type(object=outputs, check_type=dict, name='outputs')
        if self.device_num > 1: outputs['loss'] = outputs['loss'].mean()
        return outputs

    def train_forward(self, inputs, epoch):
        # main training forward
        if self.opts.do_rdrop:
            return self.train_rdrop_forward(inputs, epoch)
        else:
            return self.train_common_forward(inputs)

    def train_backward(self, loss, factor):
        '''
        Training backward
        '''
        loss = loss * factor
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def train_update(self):
        '''
        Training update
        '''
        if self.opts.max_grad_norm is not None and self.opts.max_grad_norm > 0:
            if self.use_amp:
                # before gradient clipping the optimizer parameters must be unscaled.
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(self.optimizer) if self.use_apex else self.model.parameters(),
                self.opts.max_grad_norm)
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.opts.do_ema:
            self.model_ema.update()
        self.optimizer.zero_grad()  # Reset gradients to zero
        self.running_scheduler_on_batch()

    def train_attack(self, inputs, epoch, step, factor):
        if len(self.attack_models) >= 1:
            for key, attack_model in self.attack_models.items():
                attack_model.backup_grad()
                if key == 'fgm':
                    attack_model.attack()
                    adv_outputs = self.train_forward(inputs, epoch)
                    self.train_backward(adv_outputs['loss'], factor=factor)
                    attack_model.restore()
                elif key == 'pgd':
                    for t in range(self.opts.pgd_number):
                        attack_model.attack(is_first_attack=(t == 0))
                        if t != self.opts.pgd_number - 1:
                            self.optimizer.zero_grad()
                        else:
                            attack_model.restore_grad()
                        adv_outputs = self.train_forward(inputs, epoch)
                        self.train_backward(adv_outputs['loss'], factor=factor)
                    attack_model.restore()
                elif key == 'awp':
                    current_score = self.record_tracker['result'].get(self.opts.awp_score_monitor, None)
                    kwargs = {'epoch': epoch, 'step': step, 'score': current_score}
                    if attack_model.is_attack(**kwargs):
                        for t in range(self.opts.awp_number):
                            attack_model.attack()
                            adv_outputs = self.train_forward(inputs, epoch)
                            self.optimizer.zero_grad()  # 清空梯度
                            self.train_backward(adv_outputs['loss'], factor=factor)
                        attack_model.restore()

    def train_step(self, batch, epoch, step, factor):
        batch_inputs = self.build_batch_inputs(batch)
        batch_inputs = self.process_batch_inputs(batch_inputs)
        outputs = self.train_forward(batch_inputs, epoch)
        self.train_backward(outputs['loss'], factor=factor)
        self.train_attack(batch_inputs, epoch, step, factor)
        return outputs

    def print_training_summary(self, num_examples):
        '''
        print training parameters information
        '''
        options = list(self.opts.__dict__.items())
        options.append(['num_examples', num_examples])
        options.append(['total_optimization_steps', self.num_update_training_steps])
        options.append(['total_number_of_parameters', sum(p.numel() for p in self.model.parameters())])
        options = sorted(options, key=lambda x: x[0])
        msg = '\n\n' + '=' * 10 + ' Training Start ' + '=' * 10 + '\n'
        for k, v in options:
            msg += f' - {k}: {v}\n'
        self.logger.info(msg)

    def _zero_grad(self):
        self.optimizer.zero_grad()
        self.model.zero_grad()

    def is_update(self, step):
        should_update, should_logging, should_save = False, False, False
        loss_factor = 1.0
        # Normal conditions
        if step % self.opts.gradient_accumulation_steps == 0:
            should_update = True
            self.global_step += 1
            should_logging = self.global_step % self.opts.logging_steps == 0
            should_save = self.global_step % self.opts.save_steps == 0
            loss_factor = 1.0 / self.opts.gradient_accumulation_steps
        # Each epoch save the last model, mainly for the gradient_accumulation_steps>1 case
        elif step == self.steps_in_epoch and self.opts.gradient_accumulation_steps > 1:
            should_update = True
            self.global_step += 1
            loss_factor = 1.0 / self.remaind_in_epoch
            should_logging, should_save = True, True
        else:
            pass
        return should_update, should_logging, should_save, loss_factor

    # TODO distributed training
    def train(self, train_data, dev_data=None, resume_path=None, state_to_save=None, train_with_add_datasets=None,
              convert_output_cuda_to_cpu=True):
        '''
        train function
        Args:
            train_data:
            dev_data:
            resume_path:
            state_to_save: Additional Variables need to be saved, for example {'vocab':tokenizer}
            train_with_add_dataset:Adding additional datasets, such as pseudo-labeled data
            convert_output_cuda_to_cpu: Convert cuda storage to cpu, mainly used in eval or predict phase to avoid OOM, default is True.
        Returns:
        '''
        if (
                train_with_add_datasets is not None
                and isinstance(train_data, Dataset)
        ):
            train_with_add_datasets = convert_to_list(train_with_add_datasets)
            msg = ("If dataset is not None, the dataset is added to the training data. "
                   f"The size of data : from {len(train_data)} to {sum([len(x) for x in train_with_add_datasets])}."
                   )
            self.logger.info(msg)
            for dset in train_with_add_datasets:
                train_data = ConcatDataset([train_data, dset])
                train_data.build_data_collator = dset.build_data_collator
        train_dataloader = self.build_train_dataloader(train_data)
        self.steps_in_epoch = len(train_dataloader)
        self.round_in_epoch = self.steps_in_epoch // self.opts.gradient_accumulation_steps
        self.remaind_in_epoch = self.steps_in_epoch % self.opts.gradient_accumulation_steps
        self.num_update_steps_per_epoch = max(1,
                                              self.round_in_epoch + 1 if self.remaind_in_epoch > 0 else self.round_in_epoch)
        self.num_update_training_steps = self.num_update_steps_per_epoch * self.opts.num_train_epochs
        self.scheduler = self.build_lr_scheduler()
        self.resume_from_checkpoint(resume_path=resume_path)
        self.build_model_warp()
        self.reset_metrics()
        self.build_eval_and_save_steps()
        self.build_record_tracker()
        self.print_training_summary(len(train_data))
        self._zero_grad()
        seed_everything(self.opts.seed, verbose=False)  # Added here for reproductibility (even between python 2 and 3)
        pbar = ProgressBar(n_total=self.num_update_steps_per_epoch, desc='Training',
                           num_epochs=self.opts.num_train_epochs)
        for epoch in range(1, int(self.opts.num_train_epochs) + 1):
            if self.opts.epoch_seed:
                seed_everything(self.opts.seed + epoch)  # To turn off or not, do experiment
            pbar.epoch(current_epoch=epoch)
            gc.collect()
            for step, batch in enumerate(train_dataloader):
                step += 1
                should_update, should_logging, should_save, loss_factor = self.is_update(step)
                outputs = self.train_step(batch, epoch, step, loss_factor)
                msg = {'loss': outputs['loss'].item(), "lr": self.optimizer.param_groups[0]['lr']}
                step_round = step // self.opts.gradient_accumulation_steps
                if step_round == self.round_in_epoch:
                    if step % self.opts.gradient_accumulation_steps > 0:
                        step_round = step_round + 1
                pbar.step(step=step_round, info=msg)
                if should_update:
                    self.train_update()
                    self.build_record_meter('train_loss_meter', outputs['loss'].item(), 1)
                    self.writer.add_scalar('loss/train_loss', outputs['loss'].item(), self.global_step)
                    if hasattr(self.scheduler, 'get_lr'):
                        self.writer.add_scalar('learningRate/train_lr', self.scheduler.get_lr()[0], self.global_step)
                if self.global_step > 0:
                    if should_logging:
                        train_loss = self.record_tracker['train_loss_meter'].avg
                        self.build_record_tracker()
                        if dev_data is not None:
                            # Before each eval, you need to reset the metric
                            self.reset_metrics()
                            eval_outputs = self.evaluate(self.model, dev_data,
                                                         convert_output_cuda_to_cpu=convert_output_cuda_to_cpu)
                            eval_outputs = self.process_batch_outputs(eval_outputs)
                            self.update_metrics(eval_outputs)
                            if self.opts.do_ema:
                                self.model_ema.apply_shadow()
                                self.reset_metrics()
                                eval_ema_outputs = self.evaluate(self.model, dev_data, postfix='ema',
                                                                 convert_output_cuda_to_cpu=convert_output_cuda_to_cpu)
                                eval_ema_outputs = self.process_batch_outputs(eval_ema_outputs)
                                self.update_metrics(eval_ema_outputs, postfix='ema')
                                self.model_ema.restore()
                        self.record_tracker['result']['train_loss'] = train_loss
                        self.print_evaluate_result()
                        if hasattr(self.writer, 'save'):
                            self.writer.save()
                    if should_save:
                        # model checkpoint
                        if self.model_checkpoint:
                            state = self.build_state_object(**state_to_save)
                            step_metric_score = self.name_to_metric(self.model_checkpoint.monitor)
                            self.model_checkpoint.step(
                                state=state,
                                current=step_metric_score
                            )
            self.running_scheduler_on_epoch()
            # early_stopping
            if self.early_stopping:
                step_metric_score = self.name_to_metric(self.early_stopping.monitor)
                self.early_stopping.step(current=step_metric_score)
                if self.early_stopping.stop_training:
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.opts.do_swa:
            self.model_swa = SWA(self.model_swa, self.opts.output_dir, swa_start=self.opts.swa_start)
            self.reset_metrics()
            eval_swa_outputs = self.evaluate(self.model_swa, dev_data, postfix='swa',
                                             convert_output_cuda_to_cpu=convert_output_cuda_to_cpu)
            eval_swa_outputs = self.process_batch_outputs(eval_swa_outputs)
            self.update_metrics(eval_swa_outputs, postfix='swa')
        if self.writer:
            self.writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def build_postfix(self, postfix):
        postfix = '' if postfix in [None, ''] else postfix + "_"
        return postfix

    def evaluate(self, model, dev_data, data_type='eval', save_dir=None, save_result=False, file_name=None,
                 postfix=None,
                 convert_output_cuda_to_cpu=True):
        batches = []
        postfix = self.build_postfix(postfix)
        eval_dataloader = self.build_eval_dataloader(dev_data)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(model, batch, convert_output_cuda_to_cpu=convert_output_cuda_to_cpu)
            if batch.get("loss", None):
                self.build_record_meter(f'eval_{postfix}loss_meter', batch['loss'], 1)
            batches.append(batch)
            pbar.step(step + 1)
        # 将list形式转化为dict形式
        predict_outputs = convert_tensor_list_to_dict(batches)
        if save_result:
            self.save_predict_result(predict_outputs, file_name, postfix, data_type, save_dir)
        if has_key(self.record_tracker, f'eval_{postfix}loss_meter'):
            self.record_tracker['result'][f'eval_{postfix}loss'] = self.record_tracker[f'eval_{postfix}loss_meter'].avg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return predict_outputs

    def predict(self, model, test_data, save_result=True, file_name=None, save_dir=None, postfix=None, data_type='test',
                convert_output_cuda_to_cpu=True):

        batches = []
        test_dataloader = self.build_test_dataloader(test_data)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Predicting')
        for step, batch in enumerate(test_dataloader):
            batch = self.predict_forward(model, batch, convert_output_cuda_to_cpu=convert_output_cuda_to_cpu)
            batches.append(batch)
            pbar.step(step + 1)
        predict_outputs = convert_tensor_list_to_dict(batches)
        if save_result:
            postfix = self.build_postfix(postfix)
            self.save_predict_result(predict_outputs, file_name, postfix, data_type, save_dir)
        return predict_outputs

    def predict_forward(self, model, batch, convert_output_cuda_to_cpu=True):
        batch_inputs = self.build_batch_inputs(batch)
        model.eval()
        with torch.no_grad():
            batch_outputs = model(batch_inputs)
        if batch_outputs.get('loss', None):
            batch_outputs['loss'] = batch_outputs['loss'].mean().detach().item()
        if convert_output_cuda_to_cpu:
            batch_outputs = convert_cuda_to_cpu(batch_outputs)
            batch_inputs = batch
        batch_outputs = {key: value for key, value in dict(batch_inputs, **batch_outputs).items() if
                         key not in self.keys_to_ignore_on_save_result}
        return batch_outputs

    def print_evaluate_result(self):
        '''
        打印evaluate的结果
        '''
        if len(self.record_tracker['result']) == 0:
            self.logger.warning(f"Evaluating results of {self.opts.task_name} are empty")
        if self.opts.logging_steps < self.num_update_steps_per_epoch: print(" ")
        msg = f"Result: | {self.global_step}/{self.num_update_training_steps} steps "
        for key, value in self.record_tracker['result'].items():
            if isinstance(value, (int, float)):
                name = "_".join(key.split("_")[1:]) if "_" in key else key
                self.writer.add_scalar(f"{name}/{key}", value, int(self.global_step))
                value = str(round(value, 5))
                msg += f"| {key}: {value} "
        self.logger.info(msg)

    def update_metrics(self, outputs, postfix=None):
        postfix = self.build_postfix(postfix)
        for metric in self.metrics:
            metric.update(preds=outputs['preds'], target=outputs['target'])
            value = metric.value()
            if isinstance(value, float):
                self.record_tracker['result'][f'eval_{postfix}{metric.name()}'] = value
            elif isinstance(value, dict):
                self.record_tracker['result'].update({f"eval_{postfix}{k}": v for k, v in value.items()})
            elif value is None:
                self.logger.info(f"The value of {metric.name()} is None")
            else:
                msg = "The type of metric value: expected one of (float, dict,None)"
                raise ValueError(msg)

    def save_predict_result(self, data, file_name, postfix, data_type, save_dir=None):
        '''
        保存预测信息
        '''
        if save_dir is not None:
            if not os.path.isdir(save_dir):
                save_dir = os.path.join(self.opts.output_dir, save_dir)
        else:
            save_dir = self.opts.output_dir
        if file_name is None:
            file_name = self.prefix + postfix + data_type + "_results.pkl"
        file_path = os.path.join(save_dir, file_name)
        if ".pkl" in file_path:
            save_pickle(file_path=file_path, data=data)
        elif ".json" in file_path:
            if isinstance(data, list):
                json_to_text(file_path=file_path, data=data)
            elif isinstance(data, dict):
                save_json(data=data, file_path=file_path)
            else:
                pass
        else:
            raise ValueError("file type: expected one of (.pkl, .json)")

    def process_batch_outputs(self, *args, **kwargs):
        '''
        对eval或者predict结果进行处理，适配metric计算
        Args:
            *args:
            **kwargs:
        Returns:
        '''
        raise NotImplementedError('Method [TrainBaseBuilder.process_batch_outputs] should be implemented.')
