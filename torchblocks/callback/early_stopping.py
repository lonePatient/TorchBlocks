import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping(object):
    '''
    Monitor a validation metric and stop training when it stops improving.

    Args:
        monitor: quantity to be monitored. Default: ``'eval_loss'``.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0.0``.
        patience: number of validation epochs with no improvement
            after which training will be stopped. Default: ``10`.
        verbose: verbosity mode. Default: ``True``.
        mode: one of {min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing. Default: ``'min'``.
    '''

    def __init__(self, min_delta=0, patience=10, verbose=False, mode='min', monitor='eval_loss', save_state_path=None,
                 checkpoint_state_path=None):

        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.monitor = monitor
        self.wait_count = 0
        self.stop_training = False
        self.save_state_path = save_state_path

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError("mode: expected one of (min,max)")

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        if self.verbose:
            logger.info(f'EarlyStopping mode set to {mode} for monitoring {self.monitor}.')

        if checkpoint_state_path is not None:
            self.load_state(checkpoint_state_path)
        else:
            self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf

    def save_state(self, save_path):
        state = {
            'wait_count': self.wait_count,
            'best_score': self.best_score,
            'patience': self.patience
        }
        torch.save(state, save_path)

    def load_state(self, checkpointed_state):
        state = torch.load(checkpointed_state)
        self.wait_count = state['wait_count']
        self.best_score = state['best_score']
        self.patience = state['patience']

    def step(self, current):
        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    logger.info(f"{self.patience} epochs with no improvement after which training will be stopped")
                if self.save_state_path is not None:
                    self.save_state(self.save_state_path)
