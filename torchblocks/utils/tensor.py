import torch
import numpy as np
import numbers


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
    return tensor.cpu().detach()


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
