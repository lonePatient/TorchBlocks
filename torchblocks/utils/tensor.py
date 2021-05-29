import torch
import numbers
import numpy as np

def numpy_to_tensor(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("array type: expected one of (np.ndarray)")
    return torch.from_numpy(array)


def number_to_tensor(number):
    if not isinstance(number, numbers.Number):
        raise ValueError("number type: expected one of (numbers.Number)")
    return torch.tensor([number])


def tensor_to_cpu(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor)")
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
    """
    将 Sequence length 转换成 Mask

    >>> lens = [3, 5, 4]
    >>> length_to_mask(length)
    >>> [[1, 1, 1, 0, 0],\
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 0]]

    :param length: [batch,]
    :param max_len: 最大长度
    :param dtype: nn.dtype
    :return: batch * max_len : 如果 max_len is None
    :return: batch * max(length) : 如果 max_len is None
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
        max_len = max_len or torch.max(length)
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype). \
               expand(length.shape[0], max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def pad_sequence(sequences, batch_first=True, pad_value=0):
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
