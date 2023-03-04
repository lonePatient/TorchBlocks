import torch
from .attack_base import AttackBaseBuilder

class FGM(AttackBaseBuilder):
    def __init__(self, model, attack_name, epsilon=1.0):
        super(FGM, self).__init__()
        self.model = model
        self.attack_name = attack_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.attack_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.attack_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}
