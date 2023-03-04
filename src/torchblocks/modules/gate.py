import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    """Gate Unit
    g = sigmoid(Wx)
    x = g * x
    """
    def __init__(self, input_size, dropout_rate=0.):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            res: batch * len * dim
        """
        if self.dropout_rate:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_proj = self.linear(x)
        gate = torch.sigmoid(x)
        return x_proj * gate

class GatedFeedForward(nn.Module):
    """ Feed Forward Layer with Gated Linear Unit.
        https://arxiv.org/abs/2002.05202
    """
    def __init__(self, hidden_size, feedforward_size, has_bias=True):
        super(GatedFeedForward, self).__init__()
        self.linear_gate = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = nn.ELU()

    def forward(self, x):
        gate = self.act(self.linear_gate(x))
        inter_linear = self.linear_1(x)
        inter = gate * inter_linear
        output = self.linear_2(inter)
        return output