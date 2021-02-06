import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCE, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        c = input.size()[-1]
        log_preds = F.log_softmax(input, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        loss_1 = loss * self.eps / c
        loss_2 = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss_1 + (1 - self.eps) * loss_2






