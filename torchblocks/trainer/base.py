import os
import torch
from argparse import Namespace

from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from ..optim import AdamW
from ..optim.lr_scheduler import get_linear_schedule_with_warmup

from ..utils.paths import save_pickle, json_to_text
from ..utils.tools import seed_everything, AverageMeter, to_json_string
from ..callback import ModelCheckpoint, EarlyStopping, ProgressBar, TrainLogger, EMA

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


class TrainerBase:
    def __init__(self,
                 args,
                 metrics,
                 logger,
                 batch_input_keys,
                 prefix=None,
                 collate_fn=None,
                 scheduler_on_batch=True,
                 **kwargs):  # 增加新参数

        self.args = args
        self.metrics = metrics
        self.logger = logger
        self.batch_input_keys = batch_input_keys
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
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir, self.prefix + '_tb_logs'))
        self.tb_writer.add_text("trainArgs", to_json_string(self.args.__dict__))

        # checkpoint
        self.model_checkpoint = ModelCheckpoint(
            mode=self.args.mcpt_mode,
            monitor=self.args.monitor,
            checkpoint_dir=self.args.output_dir,
            save_best_only=self.args.do_save_best
        )

        # earlystopping
        if self.args.patience <= 0:
            self.early_stopping = None
        else:
            self.early_stopping = EarlyStopping(
                patience=self.args.patience,
                mode=self.args.mcpt_mode,
                monitor=self.args.monitor
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

    def build_optimizers(self, model):
        '''
        Setup the optimizer and the learning rate scheduler.
        '''
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon,
                          weight_decay=self.args.weight_decay)
        return optimizer

    def build_scheduler(self, optimizer, t_total):
        '''
        Setup the optimizer and the learning rate scheduler.
        '''
        warmup_steps = int(t_total * self.args.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
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
        data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn)
        return data_loader

    def build_eval_dataloader(self, eval_dataset):
        '''
        Load eval dataset
        '''
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        self.logger.info("\n***** Running evaluation %s *****", self.args.task_name)
        self.logger.info("  Num examples = %d", len(eval_dataset))
        self.logger.info("  Batch size = %d", batch_size)
        sampler = SequentialSampler(eval_dataset) if self.args.local_rank == -1 else DistributedSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn)
        return data_loader

    def build_test_dataloader(self, test_dataset):
        '''
        Load test dataset
        '''
        if test_dataset is None:
            raise ValueError("Trainer: evaluation requires an test_dataset.")
        batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        self.logger.info("******** Running prediction %s ********", self.args.task_name)
        self.logger.info("  Num examples = %d", len(test_dataset))
        self.logger.info("  Batch size = %d", batch_size)
        sampler = SequentialSampler(test_dataset) if self.args.local_rank == -1 else DistributedSampler(test_dataset)
        data_loader = DataLoader(test_dataset, sampler=sampler, batch_size=batch_size, collate_fn=self.collate_fn)
        return data_loader

    def _train_step(self, model, batch, optimizer):
        '''
        Training step
        '''
        model.train()
        inputs = self.build_inputs(batch)
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return loss.item()

    def _train_update(self, model, optimizer, loss, scheduler):
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
        self.tb_writer.add_scalar('loss', loss, self.global_step)
        self.tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], self.global_step)
        # self.logger.add_value(value=loss, step=self.global_step, name='loss')
        # self.logger.add_value(value=scheduler.get_lr()[0], step=self.global_step, name="learning_rate")

    def train(self, model, train_dataset, eval_dataset):
        """
        Main training entry point.
        """
        train_dataloader = self.build_train_dataloader(train_dataset)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        optimizer = self.build_optimizers(model)
        scheduler = self.build_scheduler(optimizer, t_total)
        optimizer, scheduler = self.restore_optimizer(optimizer, scheduler)
        model, optimizer = self.prepare_for_training(model, optimizer)
        # Train!
        self.print_training_parameters(model, len(train_dataset), t_total)
        model.zero_grad()
        # ema
        if self.args.do_ema:
            ema = EMA(model, decay=self.args.ema_decay)
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 and 3)
        for epoch in range(0, int(self.args.num_train_epochs)):
            self.build_record_object()
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, batch in enumerate(train_dataloader):
                loss = self._train_step(model, batch, optimizer)
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self._train_update(model, optimizer, loss, scheduler)
                    if self.args.do_ema:
                        ema.update(model)
                    pbar(step, {'loss': loss})
                if (self.args.local_rank in [-1, 0]
                        and self.args.logging_steps > 0
                        and self.global_step % self.args.logging_steps == 0
                ):
                    if self.args.do_ema:
                        ema.apply_shadow(model)
                    self.tb_writer.add_scalar('train_epoch_loss', self.records['loss_meter'].avg,
                                              int(self.global_step / self.args.logging_steps))
                    # self.logger.add_value(value=self.records['loss_meter'].avg, step=self.global_step,
                    #                       name='train_loss')
                    print(" ")
                    self.evaluate(model, eval_dataset)
                    if self.args.do_ema:
                        ema.restore(model)
                    # log save and plot
                    # self.logger.save()
                    # save model
                if (self.args.local_rank in [-1, 0]
                        and self.args.save_steps > 0
                        and self.global_step % self.args.save_steps == 0
                ):
                    # model checkpoint
                    if self.model_checkpoint:
                        state = self.build_state_object(model, optimizer, scheduler, self.global_step)
                        self.model_checkpoint.step(
                            state=state,
                            current=self.records['result'][self.model_checkpoint.monitor],
                        )
            if not self.scheduler_on_batch:  # epoch scheduler
                scheduler.step()
            # early_stopping
            if self.early_stopping:
                self.early_stopping.step(current=self.records['result'][self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            if "cuda" in str(self.args.device):
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
        print training parameters infotmation
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
        self.logger.info("  Total Number of Parameters: %d" %
                         sum(p.numel() for p in model.parameters()))
        # Calculating total number of trainable params
        self.logger.info("  Total Number of Trainable Parameters: %d " %
                         sum(p.numel() for p in model.parameters() if p.requires_grad))

    def prepare_for_training(self, model, optimizer):
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)
        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True
            )
        return model, optimizer

    def build_inputs(self, batch):
        '''
        Sent all model inputs to the appropriate device (GPU on CPU)
        rreturn:
         The inputs are in a dictionary format
        '''
        batch = tuple(t.to(self.args.device) for t in batch)
        inputs = {key: value for key, value in zip(self.batch_input_keys, batch)}
        return inputs

    def print_evaluate_result(self):
        '''
        打印evaluation结果
        '''
        print(' ')
        if len(self.records['result']) == 0:
            self.logger.warning("eval record is empty")
        self.logger.info("\n***** Eval results of %s *****", self.args.task_name)
        self.logger.info("  global step = %s", self.global_step)
        for key in sorted(self.records['result'].keys()):
            self.logger.info("  %s = %s", key, str(self.records['result'][key]))
            self.tb_writer.add_scalar(key, self.records['result'][key], self.global_step / self.args.logging_steps)
            # self.logger.add_value(value=self.records['result'][key], step=self.global_step, name=key)

    def save_predict_result(self, file_name, data, file_dir=None):
        '''
        保存预测信息
        '''
        if file_dir is None:
            file_dir = self.args.output_dir
        file_path = os.path.join(file_dir, file_name)
        if ".pkl" in file_path:
            save_pickle(data=data, file_path=file_path)
        elif ".json" in file_path:
            json_to_text(file_path=file_path, data=data)
        else:
            raise ValueError("file type: expected one of (.pkl, .json)")

    def evaluate(self, model, eval_dataset, prefix='', save_preds=False):
        '''
        Evaluate the model on a validation set
        '''
        eval_dataloader = self.build_eval_dataloader(eval_dataset)
        self._predict_forward(model, eval_dataloader, do_eval=True)
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
                        raise ValueError("value type: expected one of (float, dict)")
                else:
                    self.logger.info(f"{metric.name()} value is None")
        if save_preds:
            output_logits_file = f"{self.prefix + prefix}_predict_eval_logits.pkl"
            self.save_predict_result(file_name=output_logits_file, data=self.records['preds'])
        self.records['result']['eval_loss'] = self.records['loss_meter'].avg
        self.print_evaluate_result()
        if 'cuda' in str(self.args.device):
            torch.cuda.empty_cache()

    def predict(self, model, test_dataset, prefix=''):
        test_dataloader = self.build_test_dataloader(test_dataset)
        self._predict_forward(model, test_dataloader, do_eval=False)
        output_logits_file = f"{self.prefix + prefix}_predict_test_logits.pkl"
        self.save_predict_result(file_name=output_logits_file, data=self.records['preds'])

    def _predict_forward(self, model, data_loader, do_eval, **kwargs):
        self.build_record_object()
        raise NotImplementedError
