import numpy as np


def softmax(x):
    assert len(x.shape) == 2
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(x - s)
    div = np.sun(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div
