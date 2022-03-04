import torch
import torch.nn as nn


def open_all_layers(model):
    r"""Open all layers in model for training.

    Examples::
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def freeze_to(n, model):
    """Freeze first n layers of model
    * **n** - Starting from initial layer, freeze all layers up to nth layer inclusively
    """
    layers = list(model.parameters())
    # Freeze up to n layers
    for param in layers[:n]:
        param.requires_grad = False
    for param in layers[n:]:
        param.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Open specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )
    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
