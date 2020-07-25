import torch
import torch.nn as nn


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, input):
        input = input.unsqueeze(2)  # (N, T, 1, K)
        input = input.permute(0, 3, 2, 1)  # (N, K, 1, T)
        input = super(SpatialDropout, self).forward(input)  # (N, K, 1, T), some features are masked
        input = input.permute(0, 3, 2, 1)  # (N, T, 1, K)
        return input.squeeze(2)  # (N, T, K)


class MultiSampleDropout(nn.Module):
    '''
    # multisample dropout (wut): https://arxiv.org/abs/1905.09788
    '''

    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits
