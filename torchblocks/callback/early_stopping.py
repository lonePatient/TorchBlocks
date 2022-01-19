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

    mode_dict = {'min': torch.lt, 'max': torch.gt}

    def __init__(self,
                 min_delta=0,
                 patience=10,
                 verbose=True,
                 mode='min',
                 monitor='eval_loss',
                 save_state_path=None,
                 load_state_path=None
                 ):

        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.monitor = monitor
        self.wait_count = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.save_state_path = save_state_path

        if mode not in self.mode_dict:
            raise ValueError(f"mode: expected one of {', '.join(self.mode_dict.keys())}")
        self.monitor_op = self.mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

        if load_state_path is not None:
            self.load_state(load_state_path)
        if self.verbose:
            logger.info(f'EarlyStopping mode set to {mode} for monitoring {self.monitor}.')

    def save_state(self, save_path):
        state = {
            'wait_count': self.wait_count,
            'best_score': self.best_score,
            'patience': self.patience
        }
        torch.save(state, save_path)

    def load_state(self, state_path):
        state = torch.load(state_path)
        self.wait_count = state['wait_count']
        self.best_score = state['best_score']
        self.patience = state['patience']

    def step(self, current):
        if not isinstance(current, torch.Tensor): current = torch.tensor(current)
        if self.monitor_op(current, self.best_score):
            msg = (
                f" Metric {self.monitor} improved from {self.best_score:.4f} to {current:.4f}"
                f" New best score: {current:.3f}"
            )
            self.best_score = current
            self.wait_count = 0
            logger.info(msg)
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    msg = (f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                           f" Best score: {self.best_score:.3f}. Signaling Trainer to stop.")
                    logger.info(msg)
                if self.save_state_path is not None:
                    self.save_state(self.save_state_path)
