import torch.nn as nn


class Application(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        raise NotImplementedError('Method [Application.forward] should be implemented.')

    def compute_loss(self, **kwargs):
        raise NotImplementedError('Method [Application.compute_loss] should be implemented.')
