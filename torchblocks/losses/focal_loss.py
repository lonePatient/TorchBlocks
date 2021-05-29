import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss
    """
    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = F.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = F.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


class FocalCosineLoss(nn.Module):
    """Implementation Focal cosine loss.

    [Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble](https://arxiv.org/abs/2007.07805).

    Source : <https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271>
    """

    def __init__(self, alpha= 1, gamma= 2, xent= 0.1, reduction="mean"):
        """Constructor for FocalCosineLoss.
        """
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, logits, target):
        """Forward Method."""
        cosine_loss = F.cosine_embedding_loss(
            logits,
            torch.nn.functional.one_hot(target, num_classes=logits.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(logits), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss
