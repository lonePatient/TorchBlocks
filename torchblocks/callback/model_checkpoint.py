import os
import torch
import logging
import numpy as np
from torchblocks.utils.paths import save_model
from torchblocks.utils.paths import json_to_text

logger = logging.getLogger(__name__)

CHECKPOINT_DIR_PREFIX = 'checkpoint'
WEIGHTS_NAME = 'pytorch_model.bin'
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
VOCAB_NAME = 'vocab.json'


class ModelCheckpoint(object):
    '''
        Save the model after every epoch by monitoring a quantity.
    args:
        monitor: quantity to monitor. Default: ``eval_loss`` ,when save_best_only=True
        save_best_only: When `True`, always saves the best score model to a file `checpoint-best`. Default: ``False``.
    '''
    mode_dict = {'min': torch.lt, 'max': torch.gt}

    def __init__(self,
                 ckpt_dir,
                 mode='min',
                 monitor='eval_loss',
                 verbose=True,
                 save_best=False,
                 keys_to_ignore_on_save=[]
                 ):
        self.ckpt_dir = ckpt_dir
        self.monitor = monitor
        self.verbose = verbose
        self.save_best = save_best
        self.keys_to_ignore_on_save = keys_to_ignore_on_save

        if mode not in self.mode_dict:
            raise ValueError(f"mode: expected one of {', '.join(self.mode_dict.keys())}")
        self.monitor_op = self.mode_dict[mode]
        torch_inf = torch.tensor(np.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.init_save_dir()

    def init_save_dir(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        prefix = f"{CHECKPOINT_DIR_PREFIX}-{self.monitor}"
        if self.save_best:
            self.save_ckpt_dir = os.path.join(self.ckpt_dir, f"{prefix}-best")
        else:
            self.save_ckpt_dir = os.path.join(self.ckpt_dir, prefix + '-{:.4f}-step-{}')

    def step(self, state, current):
        if not isinstance(current, torch.Tensor): current = torch.tensor(current)
        state['monitor'] = self.monitor
        state['score'] = current
        state['save_dir'] = self.save_ckpt_dir
        global_step = state['global_step']
        is_saving = False
        if not self.save_best:
            is_saving = True
            state['save_dir'] = self.save_ckpt_dir.format(state['score'], global_step)
        if self.monitor_op(current, self.best_score):  # best
            msg = (
                f" Steps {global_step}: Metric {self.monitor} improved from {self.best_score:.4f} to {state['score']:.4f}"
                f". New best score: {state['score']:.4f}"
            )
            logger.info(msg)
            self.best_score = current
            state['best_score'] = self.best_score
            is_saving = True
        if is_saving:
            for key in self.keys_to_ignore_on_save:
                if key in state:
                    state.pop(key)
            self.save_checkpoint(state)

    def save_checkpoint(self, state):
        os.makedirs(state['save_dir'], exist_ok=True)
        self._save_model(state)
        self._save_vocab(state)
        self._save_optimizer(state)
        self._save_scheduler(state)
        self._save_scaler(state)
        self._save_state(state)

    def _save_model(self, state):
        assert 'model' in state, "state['model'] does not exist."
        if self.verbose:
            logger.info("Saving model checkpoint to %s", state['save_dir'])
        model = state['model']
        if hasattr(model, 'save'):
            model.save(state['save_dir'])
        elif hasattr(model, 'save_pretrained'):
            model.save_pretrained(state['save_dir'])
        else:
            model_path = os.path.join(state['save_dir'], WEIGHTS_NAME)
            save_model(model, model_path)
        state.pop('model')

    def _save_vocab(self, state):
        if state.get('vocab', None):
            vocab = state['vocab']
            if hasattr(vocab, 'save_pretrained'):
                vocab.save_pretrained(state['save_dir'])
            else:
                file_path_name = os.path.join(state['save_dir'], VOCAB_NAME)
                if isinstance(vocab, dict):
                    json_to_text(file_path=file_path_name, data=vocab)
            state.pop('vocab')

    def _save_optimizer(self, state):
        if state.get('optimizer', None):
            file_path = os.path.join(state['save_dir'], OPTIMIZER_NAME)
            torch.save(state['optimizer'].state_dict(), file_path)
            state.pop('optimizer')

    def _save_scheduler(self, state):
        if state.get('scheduler', None):
            file_path = os.path.join(state['save_dir'], SCHEDULER_NAME)
            torch.save(state['scheduler'].state_dict(), file_path)
            state.pop('scheduler')

    def _save_scaler(self, state):
        if state.get('scaler', None):
            file_path = os.path.join(state['save_dir'], TRAINER_STATE_NAME)
            torch.save(state['scaler'].state_dict(), file_path)
            state.pop('scaler')

    def _save_state(self, state):
        file_path = os.path.join(state['save_dir'], TRAINER_STATE_NAME)
        torch.save(state, file_path)
