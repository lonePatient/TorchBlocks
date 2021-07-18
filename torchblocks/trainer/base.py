import os
import torch
from argparse import Namespace

from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW,get_linear_schedule_with_warmup

from torchblocks.utils.paths import save_pickle, json_to_text
from torchblocks.utils.tools import seed_everything, AverageMeter, to_json_string
from torchblocks.callback import ModelCheckpoint, EarlyStopping, ProgressBar, TrainLogger, EMA

try:
    from apex import amp
    _has_apex = True
except ImportError:
    _has_apex = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


def is_apex_available():
    return _has_apex


if not is_tensorboard_available():
    from torchblocks.callback.file_writer import FileWriter


class BaseTrainer:
    def __init__(self,
                 args,
                 metrics,
                 logger,
                 input_keys,
                 prefix=None,
                 collate_fn=None,
                 scheduler_on_batch=True,
                 **kwargs):

        self.args = args
        self.metrics = metrics
        self.logger = logger
        self.input_keys = input_keys
        self.scheduler_on_batch = scheduler_on_batch
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.global_step = 0

        if prefix is None:
            self.prefix = "_".join([self.args.model_name, self.args.task_name])

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        if not isinstance(self.args, Namespace):
            raise ValueError(
                "Parameter 'self.args'  should be an instance of class `Namespace`. "
            )

        if not isinstance(self.logger, TrainLogger):
            raise ValueError(
                "Parameter 'self.logger'  should be an instance of class `TrianLogger`. "
            )
        # tensorboard
        if is_tensorboard_available():
            self.tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f'{self.prefix}_tb_logs'))
            self.tb_writer.add_text("TrainArgs", to_json_string(args.__dict__))
        else:
            self.tb_writer = FileWriter(log_dir=os.path.join(args.output_dir, f'{self.prefix}_tb_logs'))
        # checkpoint
        self.model_checkpoint = ModelCheckpoint(
            mode=args.mcpt_mode,
            monitor=args.monitor,
            checkpoint_dir=args.output_dir,
            save_best_only=args.do_save_best
        )
        # earlystopping
        if args.patience <= 0:
            self.early_stopping = None
        else:
            self.early_stopping = EarlyStopping(
                patience=args.patience,
                mode=args.mcpt_mode,
                monitor=args.monitor
            )

    def build_record_object(self, **kwargs):
        '''
        build record object
        '''
        self.records = {}
        self.records['result'] = {}  # training result dict
        self.records['preds'] = []  # pred list
        self.records['target'] = []  # true target list
        self.records['input_lens'] = []  # input length list
        self.records['loss_meter'] = AverageMeter()

        for key, value in kwargs.items():
            if key not in self.records:
                self.records[key] = value

        for metric in self.metrics:
            metric.reset()

    def build_optimizer(self, model):
        '''
        Setup the optimizer.
        '''
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon,
                          correct_bias=self.args.correct_bias,
                          weight_decay=self.args.weight_decay)
        return optimizer

    def build_lr_scheduler(self, optimizer, t_total):
        '''
        the learning rate scheduler.
        '''
        warmup_steps = int(t_total * self.args.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        return scheduler

    def build_train_dataloader(self, train_dataset):
        '''
        Load train dataset
        '''
        if train_dataset is None:
            raise ValueError("Trainer: training requires an train_dataset.")
        batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset)
        data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn,
                                 num_workers=self.args.num_workers)
        return data_loader

    def build_eval_dataloader(self, eval_dataset):
        '''
        Load eval dataset
        '''
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        sampler = SequentialSampler(eval_dataset) if self.args.local_rank == -1 else DistributedSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn,
                                 num_workers=self.args.num_workers)
        return data_loader

    def build_test_dataloader(self, test_dataset):
        '''
        Load test dataset
        '''
        if test_dataset is None:
            raise ValueError("Trainer: evaluation requires an test_dataset.")
        batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        sampler = SequentialSampler(test_dataset) if self.args.local_rank == -1 else DistributedSampler(test_dataset)
        data_loader = DataLoader(test_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn,
                                 num_workers=self.args.num_workers)
        return data_loader

    def freeze_to(self, n, model):
        """Freeze first n layers of model
        * **n** - Starting from initial layer, freeze all layers up to nth layer inclusively
        """
        layers = list(model.parameters())
        # Freeze up to n layers
        for param in layers[:n]:
            param.requires_grad = False
        for param in layers[n:]:
            param.requires_grad = True

    def build_inputs(self, batch):
        '''
        Sent all model inputs to the appropriate device (GPU on CPU)
        rreturn:
         The inputs are in a dictionary format
        '''
        batch = tuple(t.to(self.args.device) for t in batch)
        inputs = {key: value for key, value in zip(self.input_keys, batch)}
        return inputs

    def check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def train_step(self, model, batch, optimizer):
        '''
        Training step
        '''
        model.train()
        inputs = self.build_inputs(batch)
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.check_nan(loss)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return loss.item()

    def train_update(self, model, optimizer, loss, scheduler):
        '''
        Tranining update
        '''
        if self.args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        optimizer.step()  # Update weights
        if self.scheduler_on_batch:
            scheduler.step()  # Update learning rate schedule
        model.zero_grad()  # Reset gradients to zero
        self.global_step += 1
        self.records['loss_meter'].update(loss, n=1)
        self.tb_writer.add_scalar('Loss/train_step_loss', loss, self.global_step)
        self.tb_writer.add_scalar('LearningRate/train_lr', scheduler.get_lr()[0], self.global_step)

    def train(self, model, train_dataset, eval_dataset):
        """
        Main training entry point.
        """
        train_dataloader = self.build_train_dataloader(train_dataset)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        optimizer = self.build_optimizer(model)
        scheduler = self.build_lr_scheduler(optimizer, t_total)
        optimizer, scheduler = self.restore_optimizer(optimizer, scheduler)
        model, optimizer = self.build_apex_and_distribute(model, optimizer)
        # Train!
        self.print_training_parameters(model, len(train_dataset), t_total)
        model.zero_grad()
        # ema
        if self.args.do_ema:
            ema = EMA(model, decay=self.args.ema_decay)
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 and 3)
        if self.args.logging_steps < 0:
            self.args.logging_steps = len(train_dataloader)
        if self.args.save_steps < 0:
            self.args.save_steps = len(train_dataloader)
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=self.args.num_train_epochs)
        for epoch in range(0, int(self.args.num_train_epochs)):
            self.build_record_object()
            pbar.reset()
            pbar.epoch_start(current_epoch=epoch)
            for step, batch in enumerate(train_dataloader):
                loss = self.train_step(model, batch, optimizer)
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.train_update(model, optimizer, loss, scheduler)
                    if self.args.do_ema:
                        ema.update(model)
                    pbar(step, {'loss': loss})
                if (self.args.local_rank in [-1, 0]
                        and self.args.logging_steps > 0
                        and self.global_step % self.args.logging_steps == 0
                ):
                    if self.args.do_ema:
                        ema.apply_shadow(model)
                    self.tb_writer.add_scalar('Loss/train_epoch_loss', self.records['loss_meter'].avg,
                                              int(self.global_step / self.args.logging_steps))
                    self.evaluate(model, eval_dataset)
                    if self.args.do_ema:
                        ema.restore(model)
                    if hasattr(self.tb_writer, 'save'):
                        self.tb_writer.save()
                if (self.args.local_rank in [-1, 0]
                        and self.args.save_steps > 0
                        and self.global_step % self.args.save_steps == 0
                ):
                    # model checkpoint
                    if self.model_checkpoint:
                        state = self.build_state_object(model, optimizer, scheduler, self.global_step)
                        self.model_checkpoint.step(
                            state=state,
                            current=self.records['result'][self.model_checkpoint.monitor]
                        )
            if not self.scheduler_on_batch:  # epoch scheduler
                scheduler.step()
            # early_stopping
            if self.early_stopping:
                self.early_stopping.step(current=self.records['result'][self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.tb_writer:
            self.tb_writer.close()

    def build_state_object(self, model, optimizer, scheduler, step, **kwargs):
        '''
        save state object
        '''
        state = {
            'model': model.module if hasattr(model, "module") else model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'step': step,
            'args': self.args
        }
        for key, value in kwargs.items():
            if key not in state:
                state[key] = value
        return state

    def restore_optimizer(self, optimizer, scheduler):
        '''
        Check if continuing training from a checkpoint
        '''
        if (self.args.model_path is not None
                and os.path.isfile(os.path.join(self.args.model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(self.args.model_path, "scheduler.pt"))
        ):
            self.logger.info("Load in optimizer and scheduler states from %s", self.args.model_path)
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model_path, "scheduler.pt")))
            if os.path.isfile(os.path.join(self.args.model_path, "state.bin")):
                state = torch.load(os.path.join(self.args.model_path, "state.bin"))
                if self.model_checkpoint and hasattr(state, 'best'):
                    self.model_checkpoint.best = state['best']
        return optimizer, scheduler

    def print_training_parameters(self, model, examples, t_total):
        '''
        print training parameters information
        '''
        self.logger.info("Training/evaluation parameters %s", self.args)
        self.logger.info("***** Running training %s *****", self.args.task_name)
        self.logger.info("  Model name = %s", self.args.model_name)
        self.logger.info("  Num examples = %d", examples)
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                         self.args.per_gpu_train_batch_size * self.args.n_gpu * self.args.gradient_accumulation_steps * (
                             torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)
        self.logger.info("  Total Number of Parameters: %d" % sum(p.numel() for p in model.parameters()))
        # Calculating total number of trainable params
        self.logger.info("  Total Number of Trainable Parameters: %d " % sum(
            p.numel() for p in model.parameters() if p.requires_grad))

    def build_apex_and_distribute(self, model, optimizer):
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)
        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            print("{} GPUs are available.".format(self.args.n_gpu))
            model = torch.nn.DataParallel(model)
        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True
            )
        return model, optimizer

    def print_evaluate_result(self):
        '''
        打印evaluation结果
        '''
        if len(self.records['result']) == 0:
            self.logger.warning("eval result record is empty")
        self.logger.info("***** Evaluating results of %s *****", self.args.task_name)
        self.logger.info("  global step = %s", self.global_step)
        for key in sorted(self.records['result'].keys()):
            self.logger.info("  %s = %s", key, str(round(self.records['result'][key], 5)))
            name = key.split("_")[1] if "_" in key else key
            self.tb_writer.add_scalar(f"{name[0].upper() + name[1:]}/{key}", self.records['result'][key],
                                      int(self.global_step / self.args.logging_steps))

    def save_predict_result(self, file_name, data, file_dir=None):
        '''
        保存预测信息
        '''
        if file_dir is None:
            file_dir = self.args.output_dir
        file_path = os.path.join(file_dir, file_name)
        if ".pkl" in file_path:
            save_pickle(file_path=file_path, data=data)
        elif ".json" in file_path:
            json_to_text(file_path=file_path, data=data)
        else:
            raise ValueError("file type: expected one of (.pkl, .json)")

    def evaluate(self, model, eval_dataset, prefix='', save_preds=False):
        '''
        Evaluate the model on a validation set
        '''
        eval_dataloader = self.build_eval_dataloader(eval_dataset)
        self.predict_step(model, eval_dataloader, do_eval=True)
        if self.metrics:
            for metric in self.metrics:
                metric.update(input=self.records['preds'], target=self.records['target'])
                value = metric.value()
                if value:
                    if isinstance(value, float):
                        self.records['result'][f'eval_{metric.name()}'] = value
                    elif isinstance(value, dict):
                        self.records['result'].update({f"eval_{k}": v for k, v in value.items()})
                    else:
                        raise ValueError("metric value type: expected one of (float, dict)")
                else:
                    self.logger.info(f"{metric.name()} value is None")
        if save_preds:
            output_logits_file = f"{self.prefix + prefix}_predict_eval_logits.pkl"
            self.save_predict_result(file_name=output_logits_file, data=self.records['preds'])
        self.records['result']['eval_loss'] = self.records['loss_meter'].avg
        self.print_evaluate_result()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, model, test_dataset, prefix=''):
        '''
        test数据集预测
        '''
        test_dataloader = self.build_test_dataloader(test_dataset)
        self.predict_step(model, test_dataloader, do_eval=False)
        output_logits_file = f"{self.prefix + prefix}_predict_test_logits.pkl"
        self.save_predict_result(file_name=output_logits_file, data=self.records['preds'])

    def predict_step(self, model, data_loader, do_eval, **kwargs):
        '''
        预测前向过程
        '''
        self.build_record_object()
        raise NotImplementedError('Method [predict_step] should be implemented.')
