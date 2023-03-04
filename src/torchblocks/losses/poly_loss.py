import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha=0.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        self.alpha = alpha

    def forward(self, preds, target):
        """
        Forward pass
        :param preds: tensor of shape [N, num_classes]
        :param target: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(target, num_classes=self.num_classes).to(device=preds.device,
                                                                           dtype=preds.dtype)
        pt = torch.sum(labels_onehot * F.softmax(preds, dim=-1), dim=-1)
        CE = F.cross_entropy(input=preds,
                             target=target,
                             reduction='none',
                             weight=self.weight)
        poly2 = CE + self.epsilon * (1 - pt) + self.alpha * (1 - pt) * (1 - pt)
        if self.reduction == "mean":
            poly2 = poly2.mean()
        elif self.reduction == "sum":
            poly2 = poly2.sum()
        return poly2


class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot

    def forward(self, preds, target):
        """
        Forward pass
        :param preds: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param target: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        p = torch.sigmoid(preds)
        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if target.ndim == 1:
                target = F.one_hot(target, num_classes=self.num_classes)
            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                target = F.one_hot(target.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)
        target = target.to(device=preds.device, dtype=target.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=preds,
                                                     target=target,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = target * p + (1 - target) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            FL = alpha_t * FL
        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1
