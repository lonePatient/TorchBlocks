import torch
from .attack_base import AttackBaseBuilder


class PGD(AttackBaseBuilder):
    def __init__(self, model, attack_name, epsilon=1., alpha=0.3):
        super(PGD, self).__init__()
        self.model = model
        self.attack_name = attack_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.attack_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.attack_name in name:
                if is_first_attack:
                    self.attack_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.attack_name in name:
                assert name in self.attack_backup
                param.data = self.attack_backup[name]
        self.attack_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.attack_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.attack_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name or self.attack_name in name:
                    self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name or self.attack_name in name:
                    param.grad = self.grad_backup[name]
