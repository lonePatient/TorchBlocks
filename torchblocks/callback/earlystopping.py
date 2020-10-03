import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping(object):
    '''
    Stop training when a monitored quantity has stopped improving.
    '''
    def __init__(self,
                 min_delta=0,
                 patience=10,  # Interval (number of epochs) between checkpoints
                 verbose=1,
                 mode='min',
                 monitor='eval_loss',
                 baseline=None):
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.monitor = monitor
        assert mode in ['min', 'max']
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self.reset()

    def reset(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stop_training = False
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def step(self, current):
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    logger.info(f"{self.patience} epochs with no improvement after which training will be stopped")
                self.stop_training = True
