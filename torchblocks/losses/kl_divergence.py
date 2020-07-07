import torch
import torch.nn as nn
import torch.nn.functional as F


class KL(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KL, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')
        return loss


class SKL(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SKL, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        logit = input.view(-1, input.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + self.epsilon) - 1 + self.epsilon).detach().log()
        ry = -(1.0 / (y + self.epsilon) - 1 + self.epsilon).detach().log()
        return (p * (rp - ry) * 2).sum() / bs
