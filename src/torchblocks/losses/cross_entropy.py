import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, ignore_index=-100):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.weights is None:
            self.weights = torch.ones(input.shape[-1])
        self.weights = self.weights.to(input.device)
        mask = (target != self.ignore_index).any(axis=-1)
        p = F.log_softmax(input[mask], -1, dtype=input.dtype)
        w_labels = self.weights * target[mask]
        loss = -(w_labels * p).sum() / (w_labels).sum()
        return loss


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
