import numpy as np
import warnings
from typing import *

def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = 'sample', 'label'
    if average == 'samples':
        axis0, axis1 = axis1, axis0
    msg = ('{0} ill-defined and being set to 0.0 {{0}} '
           'no {1} {2}s. Use `zero_division` parameter to control'
           ' this behavior.'.format(msg_start, modifier, axis0))
    if result_size == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    warnings.warn(msg, UserWarning, stacklevel=2)


def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division='warn'):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator
    if not np.any(mask):
        return result
    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0
    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != 'warn' or metric not in warn_for:
        return result
    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."
    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result
    _warn_prf(average, modifier, msg_start, len(result))
    return result


def check_consistent_length(y_true: List[List[str]], y_pred: List[List[str]]):
    """Check that all arrays have consistent first and second dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Args:
        y_true : 2d array.
        y_pred : 2d array.
    """
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))
    if not is_list == {list}:
        raise TypeError('Found input variables without list of list.')

    if len(y_true) != len(y_pred) or len_true != len_pred:
        message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_true, len_pred)
        raise ValueError(message)