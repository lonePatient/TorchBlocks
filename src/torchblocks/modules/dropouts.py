import torch
import torch.nn as nn
from itertools import repeat

class SpatialDropout(nn.Module):
    """
    对字级别的向量进行丢弃
    """
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob
    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))
    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output


class MultiSampleDropout(nn.Module):
    '''
    # multisample dropout (wut): https://arxiv.org/abs/1905.09788
    '''

    def __init__(self, hidden_size, num_labels, K=5, dropout_rate=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits


class TimestepDropout(torch.nn.Dropout):
    r"""
    传入参数的shape为 ``(batch_size, num_timesteps, embedding_dim)``
    使用同一个shape为 ``(batch_size, embedding_dim)`` 的mask在每个timestamp上做dropout。
    """

    def forward(self, x):
        dropout_mask = x.new_ones(x.shape[0], x.shape[-1])
        torch.nn.functional.dropout(dropout_mask, self.p, self.training, inplace=True)
        dropout_mask = dropout_mask.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        if self.inplace:
            x *= dropout_mask
            return
        else:
            return x * dropout_mask

class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)

class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)