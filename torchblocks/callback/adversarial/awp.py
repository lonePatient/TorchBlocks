import torch

class AWP(object):
    """ [Adversarial weight perturbation helps robust generalization](https://arxiv.org/abs/2004.05884)
    """
    def __init__(
        self,
        model,
        emb_name="weight",
        epsilon=0.001,
        alpha=1.0,
    ):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.param_backup = {}
        self.param_backup_eps = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        if self.alpha == 0: return
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                # save
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                    grad_eps = self.epsilon * param.abs().detach()
                    self.param_backup_eps[name] = (
                        self.param_backup[name] - grad_eps,
                        self.param_backup[name] + grad_eps,
                    )
                # attack
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.alpha * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, 
                            self.param_backup_eps[name][0]
                        ), 
                        self.param_backup_eps[name][1]
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.param_backup:
                param.data = self.param_backup[name]
        self.param_backup = {}
        self.param_backup_eps = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}
