import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, preds, target):
        loss = torch.sum(-target * F.log_softmax(preds, dim=-1), dim=-1)
        return loss.mean()


class MultiLabelCategoricalCrossEntropy(nn.Module):
    """
    https://kexue.fm/archives/7359
    """
    def __init__(self):
        super(MultiLabelCategoricalCrossEntropy, self).__init__()

    def forward(self, preds, target):
        preds = (1 - 2 * target) * preds  # -1 -> pos classes, 1 -> neg classes
        preds_neg = preds - target * 1e12  # mask the pred outputs of pos classes
        preds_pos = (preds - (1 - target) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(preds[..., :1])
        preds_neg = torch.cat([preds_neg, zeros], dim=-1)
        preds_pos = torch.cat([preds_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(preds_neg, dim=-1)
        pos_loss = torch.logsumexp(preds_pos, dim=-1)
        return (neg_loss + pos_loss).mean()
