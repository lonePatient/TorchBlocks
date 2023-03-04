import torch
import torch.nn as nn
import torch.nn.functional as F


class KL(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KL, self).__init__()
        self.reduction = reduction

    def forward(self, preds, target):
        preds = preds.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(preds, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')
        return loss


class BKL(nn.Module):
    def __init__(self, reduction='mean'):
        super(BKL, self).__init__()
        self.reduction = reduction

    def forward(self, preds, target):
        preds = preds.float()
        target = target.float()
        loss1 = F.kl_div(F.log_softmax(preds, dim=-1), F.softmax(target, dim=-1), reduction=self.reduction)
        loss2 = F.kl_div(F.log_softmax(target, dim=-1), F.softmax(preds, dim=-1), reduction=self.reduction)
        loss = (loss1.mean() + loss2.mean()) / 2
        return loss


class SKL(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SKL, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, target):
        logit = preds.view(-1, preds.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + self.epsilon) - 1 + self.epsilon).detach().log()
        ry = -(1.0 / (y + self.epsilon) - 1 + self.epsilon).detach().log()
        return (p * (rp - ry) * 2).sum() / bs
