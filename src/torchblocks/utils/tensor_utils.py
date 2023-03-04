import torch
import numbers
import logging
import numpy as np
import torch.nn.functional as F
from .common_utils import check_object_type

logger = logging.getLogger()


def convert_to_tensor(obj, device=torch.device("cpu")):
    """
    Converts to Tensor if given object is not a Tensor.
    """
    if not isinstance(obj, torch.Tensor):
        obj = torch.Tensor(obj).to(device)
    return obj


def numpy_to_tensor(array, device=torch.device("cpu")):
    """
    Converts to Tensor if given object is a array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("array type: expected one of (np.ndarray,)")
    return torch.from_numpy(array).to(device)


def number_to_tensor(number, device=torch.device("cpu")):
    """
    Converts to Tensor if given object is a number.
    """
    if not isinstance(number, numbers.Number):
        raise ValueError("number type: expected one of (numbers.Number,)")
    return torch.tensor([number], device=device)


def tensor_to_cpu(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor,)")
    return tensor.detach().cpu()


def tensor_to_numpy(tensor):
    _tensor = tensor_to_cpu(tensor)
    return _tensor.numpy()


def tensor_to_list(tensor):
    _tensor = tensor_to_numpy(tensor)
    return _tensor.tolist()


def select_logits_with_mask(logits, mask):
    if len(logits.shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(logits).to(torch.bool)
        logits_select = torch.masked_select(logits, mask).view(-1, logits.size(-1))
    else:
        logits_select = logits  # Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
    return logits_select


def length_to_mask(length, max_len=None, dtype=None):
    '''
    将 Sequence length 转换成 Mask
    Args:
        length: [batch,]
        max_len: 最大长度
        dtype: nn.dtype
    Returns:
        batch * max_len : 如果 max_len is None
    Examples:
        >>> lens = [3, 5, 4]
        >>> length_to_mask(length)
        >>> [[1, 1, 1, 0, 0],\
            [1, 1, 1, 1, 1], \
            [1, 1, 1, 1, 0]]
    '''
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
        max_len = max_len or torch.max(length)
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype). \
               expand(length.shape[0], max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def pad_sequence(sequences, batch_first=True, pad_value=0):
    '''
    Args:
        sequences:
        batch_first:
        pad_value:
    Returns:
    '''

    def length(sequence):
        if isinstance(sequence, torch.Tensor):
            return sequence.size(0)
        if isinstance(sequence, list):
            return len(sequence)

    lengths, sequences = zip(*[(length(sequence), torch.as_tensor(sequence)) for sequence in sequences])
    return torch.as_tensor(lengths), torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=batch_first, padding_value=pad_value)


def to_onehot(tensor, num_classes=None):
    """
    Converts a dense label tensor to one-hot format
    Args:
        tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C
    Output:
        A sparse label tensor with shape [N, C, d1, d2, ...]
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    """
    if num_classes is None:
        num_classes = int(tensor.max().detach().item() + 1)
    dtype, device, shape = tensor.dtype, tensor.device, tensor.shape
    tensor_onehot = torch.zeros(shape[0], num_classes, *shape[1:],
                                dtype=dtype, device=device)
    index = tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def to_categorical(tensor, argmax_dim=1):
    """
    Converts a tensor of probabilities to a dense label tensor
    Args:
        tensor: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply
    Return:
        A tensor with categorical labels [N, d2, ...]
    Example:
        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])
    """
    return torch.argmax(tensor, dim=argmax_dim)


def build_dropout_mask(drop_p, tensor):
    r"""
    根据tensor的形状，生成一个mask
    :param drop_p: float, 以多大的概率置为0。
    :param tensor: torch.Tensor
    :return: torch.FloatTensor. 与tensor一样的shape
    """
    mask_x = torch.ones_like(tensor)
    F.dropout(mask_x, p=drop_p, training=False, inplace=True)
    return mask_x


def select_topk(tensor, topk=1, dim=1):
    """
    Convert a probability tensor to binary by selecting top-k highest entries.
    """
    check_object_type(object=tensor, check_type=torch.Tensor, name='tensor')
    zeros = torch.zeros_like(tensor)
    if topk == 1:  # argmax has better performance than topk
        topk_tensor = zeros.scatter(dim, tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor = zeros.scatter(dim, tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()


def concat_tensors_with_padding(tensor_list, padding_shape, padding_index=1, padding_value=0):
    """Concatenate the list of tensors to be a single tensor with paddings.

    Args:
        tensor_list: The list of tensors which have different lengths. They should have
            the shape of `(batch_size, seq_len, dim)` or `(batch_size, seq_len)`.
        padding: The padding value for the tensors. If the tensor is shorter than other
            tensors, than it will be padded with this value. Default is `0`.

    Returns:
        A concatenated single tnesor.
    """
    max_length = max(x.size(padding_index) for x in tensor_list)
    padded_tensor_list = []
    for tensor in tensor_list:
        padding_length = max_length - tensor.size(padding_index)
        padding_size = [x * padding_length for x in padding_shape]
        padded_tensor_list.append(F.pad(tensor, padding_size, value=padding_value))
    return torch.cat(padded_tensor_list)


def convert_tensor_list_to_dict(tensor_list):
    '''
    将tensor列表转化dict形式
    Args:
        tensor_list:
    Returns:
    '''
    tensor_dict = {}
    first = tensor_list[0]
    for k, v in first.items():
        bv = [f[k] for f in tensor_list]
        tensor_dict[k] = bv
    return tensor_dict

def convert_cuda_to_cpu(inputs):
    outputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            outputs[key] = tensor_to_cpu(value)
        elif isinstance(value, (list, tuple)):
            outputs[key] = [tensor_to_cpu(x) if isinstance(x, torch.Tensor) else x for x in value]
        elif isinstance(value, dict):
            outputs[key] = {x: tensor_to_cpu(y) if isinstance(y, torch.Tensor) else y for x, y in value.items()}
        else:
            outputs[key] = value
    return outputs