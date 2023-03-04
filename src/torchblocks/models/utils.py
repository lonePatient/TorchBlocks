import torch
import torch.nn as nn


def open_all_layers(model):
    r"""Open all modules in model for training.

    Examples::
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def freeze_topK(n, model):
    """Freeze first n modules of model
    * **n** - Starting from initial layer, freeze all modules up to nth layer inclusively
    """
    layers = list(model.parameters())
    # Freeze up to n modules
    for param in layers[:n]:
        param.requires_grad = False
    for param in layers[n:]:
        param.requires_grad = True


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def reinit_last_layers(model, num_layers):
    """Re-initialize the last-k transformer modules.

    Args:
        model: The target transformer model.
        num_layers: The number of modules to be re-initialized.
    """
    if num_layers > 0:
        base_model = getattr(model, model.base_model_prefix)
        base_model.encoder.layer[-num_layers:].apply(model._init_weights)


def get_parameter_groups(module):
    """Get parameter groups for transformer training.

    It is well-known that excluding layer-norm and bias parameters from weight-decay
    leads better performance at training transformer-based models. To achieve that, this
    function creates the separated parameter groups for applying weight-decay and
    ignoring weight-decay.

    Args:
        module: The target module to get the parameters from.

    Returns:
        A list of two parameter groups.
    """
    do_decay = [p for p in module.parameters() if p.ndim < 2]
    no_decay = [p for p in module.parameters() if p.ndim >= 2]
    return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]


def open_specified_layers(model, open_layers):
    r"""Open specified modules in model for training while keeping
    other modules frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): modules open for training.

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
            unfreeze(module)
        else:
            module.eval()
            freeze(module)
