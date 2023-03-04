
import torch
from torch.optim.optimizer import Optimizer, required


class AdaiW(Optimizer):
    r"""Implements Adai with decoupled weight decay (AdaiW).
    It is proposed in the ICML2022 paper  
    `Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum`
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): beta0 and beta2 (default: (0.1, 0.99))
        eps (float, optional): the inertia bound (default: 1e-03)
        weight_decay (float, optional): weight decay (default: 0)

    """

    def __init__(self, params, lr=required, betas=(0.1, 0.99), eps=1e-03,
                 weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0]:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaiW, self).__init__(params, defaults)
    

    def __setstate__(self, state):
        super(AdaiW, self).__setstate__(state)
            
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        param_size = 0
        exp_avg_sq_hat_sum = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_size += p.numel()
                grad = p.grad.data
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Cumulative products of beta1
                    state['beta1_prod'] = torch.ones_like(p.data, memory_format=torch.preserve_format)

                exp_avg_sq = state['exp_avg_sq']
                beta0, beta2 = group['betas']

                state['step'] += 1
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum()
                
        # Calculate the mean of all elements in exp_avg_sq_hat
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # Perform stable/decoupled weight decay
                if group['weight_decay'] !=0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta0, beta2 = group['betas']
                beta1_prod = state['beta1_prod']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                beta1 = (1. - (exp_avg_sq_hat / exp_avg_sq_hat_mean).mul(beta0)).clamp(0., 1 - group['eps'])
                
                beta1_prod.mul_(beta1)
                bias_correction1 = 1 - beta1_prod
                
                exp_avg.mul_(beta1).addcmul_(1 - beta1, grad)
                exp_avg_hat = exp_avg.div(bias_correction1) 
                
                p.data.add_(exp_avg_hat, alpha=-group['lr'])

        return loss
