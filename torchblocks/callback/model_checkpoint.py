import os
import torch
import logging
import numpy as np
from torchblocks.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_SAVE_MODEL_NAME = 'checkpoint'


class ModelCheckpoint(object):
    '''
    Save the model after every epoch by monitoring a quantity.
    args:
    monitor: quantity to monitor. Default: ``eval_loss`` ,when save_best_only=True
    save_best_only: When `True`, always saves the best score model to a file `checpoint-best`. Default: ``False``.
    '''

    def __init__(self, checkpoint_dir, monitor='eval_loss', mode='min', save_best_only=False, verbose=True):

        ensure_dir(checkpoint_dir)
        self.monitor = monitor
        self.verbose = verbose
        self.base_path = checkpoint_dir
        self.save_best_only = save_best_only

        if save_best_only:
            if mode == 'min':
                self.monitor_op = np.less
                self.best_score = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best_score = -np.Inf
            else:
                raise ValueError("mode: expected one of (min,max)")
            self.output_dir = os.path.join(checkpoint_dir, f"{DEFAULT_SAVE_MODEL_NAME}-best")
        else:
            self.output_dir = os.path.join(checkpoint_dir, f"{DEFAULT_SAVE_MODEL_NAME}-%s")

    def save_checkpoint(self, state, save_dir):
        ensure_dir(save_dir)
        assert 'model' in state, "state['model'] does not exist."
        if self.verbose:
            logger.info("Saving model checkpoint to %s", save_dir)
        model = state['model']

        if hasattr(model, 'save'):
            model.save(save_dir)
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(save_dir)
        state.pop('model')

        torch.save(state['args'], os.path.join(save_dir, "training_args.bin"))
        state.pop('args')

        if state.get('optimizer', None):
            if self.verbose:
                logger.info("Saving optimizer and scheduler states to %s", save_dir)
            torch.save(state['optimizer'].state_dict(), os.path.join(save_dir, "optimizer.pt"))
            state.pop('optimizer')

        if state.get('scheduler', None):
            torch.save(state['scheduler'].state_dict(), os.path.join(save_dir, "scheduler.pt"))
            state.pop('scheduler')
        if self.verbose:
            logger.info("Saving states to %s", save_dir)
        torch.save(state, os.path.join(save_dir, "state.bin"))

    def step(self, state, current):
        if self.save_best_only:
            if self.monitor_op(current, self.best_score):
                if self.verbose:
                    logger.info(
                        f" Steps {state['step']}: {self.monitor} improved from {self.best_score:.5f} to {current:.5f}")
                self.best_score = current
                state['best_score'] = self.best_score
                self.save_checkpoint(state, self.output_dir)
        else:
            output_dir = self.output_dir % state['step']
            if self.verbose:
                logger.info(f" Step {state['step']} - {self.monitor}: {current:.5f} save model to disk.")
            self.save_checkpoint(state, output_dir)
