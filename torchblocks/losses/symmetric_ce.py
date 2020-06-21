import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    '''
    paper:"Symmetric Cross Entropy for Robust Learning with Noisy Labels"
    '''
    def __init__(self, alpha, beta, num_labels):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_labels = num_labels
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # CCE
        ce = self.cross_entropy(input, target)
        # RCE
        pred = F.softmax(input, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(target, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss