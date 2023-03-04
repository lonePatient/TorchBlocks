import torch
from torch import nn

try:
    from apex.normalization import FusedLayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm as FusedLayerNorm


class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_shape, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)
        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)
        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)
        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)
        outputs = outputs / std  # (b, s, h)
        outputs = outputs * weight + bias
        return outputs


def replace_with_fused_layernorm(module):
    """Replace the normal (PyTorch-vanilla) layer-norms to apex fused layer-norms.
    Args:
        module: The target module to be replaced.
    """
    for submodule in module.modules():
        for name, layer in submodule.named_children():
            if not isinstance(layer, nn.LayerNorm):
                continue
            # Create new fused layer-norm and copy the original parameters.
            new_layer = FusedLayerNorm(layer.normalized_shape, layer.eps)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            # Replace the layer-norm to the new one.
            setattr(submodule, name, new_layer)
