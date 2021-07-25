
import torch
from torchblocks.metrics.base import Metric
from torch.nn import functional as F


class MSE(Metric):
    '''
    Computes mean squared error
       Example:
        >>> mse = MSE(reduction='mean')
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mse.update(x, y)
        >>> mse.value()
        tensor(0.2500)
    '''

    def __init__(self, reduction='none'):
        self.reduction = reduction

    def update(self, input, target):
        self.y_pred = input
        self.y_true = target

    def value(self):
        v = F.mse_loss(self.y_pred, self.y_true, reduction=self.reduction)
        return v.item()

    def name(self):
        return "mse"


class RMSE(Metric):
    '''
    Computes root mean squared error
       Example:
        >>> rmse = RMSE(reduction='mean')
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> rmse.update(x, y)
        >>> rmse.value()
        tensor(0.5000)
    '''

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def update(self, input, target):
        self.y_pred = input
        self.y_true = target

    def value(self):
        mean_squared_error = F.mse_loss(self.y_pred, self.y_true, reduction=self.reduction)
        v = torch.sqrt(mean_squared_error)
        return v.item()

    def name(self):
        return "rmse"


class MAE(Metric):
    '''
    Computes mean absolute error
       Example:
        >>> mae = MAE(reduction='mean')
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mae.update(x, y)
        >>> mae.value()
        tensor(0.2500)
    '''

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def update(self, input, target):
        if input.shape != target.shape:
            raise RuntimeError('Predictions and targets are expected to have the same shape')

        self.y_pred = input
        self.y_true = target

    def value(self):
        v = F.l1_loss(self.y_pred, self.y_true, reduction=self.reduction)
        return v.item()

    def name(self):
        return "mae"


