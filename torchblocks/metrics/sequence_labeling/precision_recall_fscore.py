import numpy as np
from typing import *
from collections import defaultdict
from torchblocks.metrics.sequence_labeling.scheme import get_scheme
from torchblocks.metrics.sequence_labeling.util import _prf_divide,_warn_prf,check_consistent_length

PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]


def _precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                     y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                     *,
                                     average: Optional[str] = None,
                                     warn_for=('precision', 'recall', 'f-score'),
                                     beta: float = 1.0,
                                     sample_weight: Optional[List[int]] = None,
                                     zero_division: str = 'warn',
                                     scheme: Optional[Type[Any]] = None,
                                     suffix: bool = False,
                                     extract_tp_actual_correct: Callable = None) -> SCORES:
    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options:
        raise ValueError('average has to be one of {}'.format(average_options))

    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred, suffix, scheme)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == 'warn' and ('f-score',) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, 'true nor predicted', 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum))

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None
    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)

    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    *,
                                    average: Optional[str] = None,
                                    labels: Optional[List[str]] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False,
                                    schema: str = "BIO",
                                    ) -> SCORES:
    """Compute precision, recall, F-measure and support for each class.
    Args:
        target : 2d array. Ground truth (correct) target values.
        preds : 2d array. Estimated targets as returned by a tagger.
        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.
        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both
            If set to "warn", this acts as 0, but warnings are also raised.
        suffix : bool, False by default.
    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]
        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]
        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]
        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.
    Examples:
        >>> from torchblocks.metrics.sequence_labeling.precision_recall_fscore import precision_recall_fscore_support
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
        (0.5, 0.5, 0.5, 2)
        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:
        >>> precision_recall_fscore_support(y_true, y_pred, average=None)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))
    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """

    def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        if len(y_pred[0]) > 0 and isinstance(y_pred[0][0], str):
            check_consistent_length(y_true, y_pred)
            get_entities = get_scheme(scheme_type=schema)
            for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
                for type_name, start, end in get_entities(y_t):
                    entities_true[type_name].add((i, start, end))
                for type_name, start, end in get_entities(y_p):
                    entities_pred[type_name].add((i, start, end))
        else:
            for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
                for start, end, type_name in y_t:
                    entities_true[type_name].add((i, start, end))
                for start, end, type_name in y_p:
                    entities_pred[type_name].add((i, start, end))
        if labels is not None:
            entities_true = {k: v for k, v in entities_true.items() if k in labels}
            entities_pred = {k: v for k, v in entities_pred.items() if k in labels}
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))
        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )
    return precision, recall, f_score, true_sum
