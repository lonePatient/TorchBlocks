import torch
import torch.nn as nn


class HardMining(nn.Module):
    def __init__(self, save_rate=2):
        super(HardMining, self).__init__()
        self.save_rate = save_rate
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, target):
        batch_size = preds.shape[0]
        loss = self.ce(preds, target)
        ind_sorted = torch.argsort(-loss)  # from big to small
        num_saved = int(self.save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(self.ce(preds[ind_update], target[ind_update]))
        return loss_final
