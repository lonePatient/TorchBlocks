import torch.nn.functional as F
import torch
import torch.nn as nn

class SymmetricCE(nn.Module):
    """Pytorch Implementation of Symmetric Cross Entropy.

    Paper: https://arxiv.org/abs/1908.06112
    """
    def __init__(self, num_classes, alpha: float = 1.0, beta: float = 1.0):
        """Constructor method for symmetric CE.

        Args:
            alpha: The alpha value for symmetricCE.
            beta: The beta value for symmetricCE.
            num_classes: The number of classes.
        """
        super(SymmetricCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        """Forward method."""
        ce = self.ce(input, target)

        logits = F.softmax(input, dim=1)
        logits = torch.clamp(logits, min=1e-7, max=1.0)
        if logits.is_cuda:
            label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().cuda()
        else:
            label_one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(logits * torch.log(label_one_hot), dim=1)
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss