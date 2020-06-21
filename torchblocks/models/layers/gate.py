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
