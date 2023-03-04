import torch
from .attack_base import AttackBaseBuilder


class AWP(AttackBaseBuilder):
    mode_dict = {'min': torch.lt, 'max': torch.gt}

    def __init__(self, model, attack_name="weight", alpha=1.0, epsilon=0.01, start_epoch=-1, start_step=-1,
                 start_score=-1, score_mode='min'):
        super(AWP, self).__init__()
        self.model = model
        self.attack_name = attack_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.backup = {}
        self.backup_eps = {}
        self.start_epoch = start_epoch
        self.start_step = start_step
        self.start_score = start_score
        self.score_mode = score_mode

    def attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                    param.requires_grad
                    and param.grad is not None
                    and self.attack_name in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.alpha * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if (
                    param.requires_grad
                    and param.grad is not None
                    and self.attack_name in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.epsilon * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def is_attack(self, **kwargs):
        epoch = kwargs.pop('epoch', 0)
        step = kwargs.pop('step', 0)
        current_score = kwargs.pop('score', 0)
        attack_start = True
        if self.start_epoch > 0:
            attack_start = False
            if epoch >= self.start_epoch:
                attack_start = True
        elif self.start_step > 0:
            attack_start = False
            if step >= self.start_step:
                attack_start = True
        elif self.start_score > 0:
            attack_start = False
            monitor_op = self.mode_dict[self.score_mode]
            if current_score is None:
                attack_start = False
            else:
                if monitor_op(current_score, self.start_score):
                    attack_start = True
        return attack_start
