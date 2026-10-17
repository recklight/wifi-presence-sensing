"""Classification and localization metrics.

Accuracy alone hides what matters for occupancy sensing: a detector that
always answers "empty" scores well on a mostly empty building and is useless.
:func:`classification_metrics` therefore reports the unweighted average recall
(UAR, the mean of per-class recalls, insensitive to class imbalance), the
per-class sensitivity and specificity, macro precision/recall/F1, the
confusion matrix and, when scores are available, ROC-AUC (Fawcett, 2006).

For localization a single threshold is equally blind; :func:`summarize_errors`
gives mean, median, 90th percentile, RMSE and maximum error together.

References
----------
T. Fawcett, "An introduction to ROC analysis," Pattern Recognition Letters,
27(8):861-874, 2006.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True, eq=False)
class ClassificationMetrics:
    """Classification performance.

    Attributes
    ----------
    labels : list of str
        Class order of the confusion matrix and the per-class entries.
    n_samples : int
        Number of evaluated samples.
    accuracy : float
        Fraction correct.
    uar : float
        Unweighted average recall (balanced accuracy).
    macro_precision, macro_recall, macro_f1 : float
        Unweighted means over classes.
    sensitivity, specificity : dict
        One-vs-rest recall of each class and of its complement.
    confusion : numpy.ndarray
        Rows are true classes, columns predicted classes.
    roc_auc : float or None
        Binary: AUC of the positive class.  Multiclass: macro one-vs-rest AUC.
        ``None`` when no scores were given or a class is absent.
    positive_label : str or None
        Class treated as positive.
    """

    labels: list[str]
    n_samples: int
    accuracy: float
    uar: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    sensitivity: dict[str, float]
    specificity: dict[str, float]
    confusion: np.ndarray
    roc_auc: float | None = None
    positive_label: str | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready representation."""
        payload = asdict(self)
        payload["confusion"] = self.confusion.tolist()
        return payload

    def headline(self) -> dict[str, float | None]:
        """The scalar metrics only."""
        result: dict[str, float | None] = {
            "accuracy": self.accuracy,
            "uar": self.uar,
            "macro_f1": self.macro_f1,
            "roc_auc": self.roc_auc,
        }
        if self.positive_label is not None and self.positive_label in self.sensitivity:
            result["sensitivity"] = self.sensitivity[self.positive_label]
            result["specificity"] = self.specificity[self.positive_label]
        return result


def sensitivity_specificity(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-vs-rest sensitivity and specificity of every class.

    Parameters
    ----------
    confusion : numpy.ndarray
        Square confusion matrix, rows true, columns predicted.

    Returns
    -------
    sensitivity, specificity : numpy.ndarray
        One value per class; NaN when undefined (no positives / negatives).

    Examples
    --------
    >>> sens, spec = sensitivity_specificity(np.array([[8, 2], [1, 9]]))
    >>> sens.tolist(), spec.tolist()
    ([0.8, 0.9], [0.9, 0.8])
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    total = matrix.sum()
    true_positive = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    false_positive = predicted - true_positive
    actual_negative = total - actual
    true_negative = actual_negative - false_positive
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.where(actual > 0, true_positive / actual, np.nan)
        specificity = np.where(actual_negative > 0, true_negative / actual_negative, np.nan)
    return sensitivity, specificity


def classification_metrics(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    labels: Sequence[object] | None = None,
    scores: np.ndarray | None = None,
    positive_label: object | None = None,
) -> ClassificationMetrics:
    """Score predicted labels with every metric of :class:`ClassificationMetrics`.

    Parameters
    ----------
    y_true, y_pred : sequence
        True and predicted labels.
    labels : sequence, optional
        Class order; defaults to the sorted union of both inputs.
    scores : numpy.ndarray, optional
        Class probabilities ``(n_samples, n_classes)`` in ``labels`` order.
    positive_label : optional
        Class treated as positive for the headline sensitivity/specificity.

    Returns
    -------
    ClassificationMetrics
        The metrics.

    Examples
    --------
    >>> m = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    >>> m.accuracy, m.uar, m.confusion.tolist()
    (0.75, 0.75, [[1, 1], [0, 2]])
    """
    truth = np.asarray([str(v) for v in y_true])
    predicted = np.asarray([str(v) for v in y_pred])
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if truth.size == 0:
        raise ValueError("cannot score an empty prediction set")
    order = (
        [str(v) for v in labels]
        if labels is not None
        else sorted(set(truth.tolist()) | set(predicted.tolist()))
    )
    confusion = confusion_matrix(truth, predicted, labels=order)
    sensitivity, specificity = sensitivity_specificity(confusion)
    present = confusion.sum(axis=1) > 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=order, average="macro", zero_division=0
    )

    roc_auc: float | None = None
    if scores is not None:
        probabilities = np.asarray(scores, dtype=np.float64)
        if probabilities.shape != (truth.size, len(order)):
            raise ValueError("scores must be (n_samples, n_classes) in label order")
        if present.all() and len(order) >= 2:
            if len(order) == 2:
                roc_auc = float(roc_auc_score(truth == order[1], probabilities[:, 1]))
            else:
                normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
                roc_auc = float(
                    roc_auc_score(
                        truth, normalized, labels=order, multi_class="ovr", average="macro"
                    )
                )

    positive = None if positive_label is None else str(positive_label)
    return ClassificationMetrics(
        labels=list(order),
        n_samples=int(truth.size),
        accuracy=float(np.mean(truth == predicted)),
        uar=float(np.nanmean(sensitivity[present])) if present.any() else float("nan"),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        sensitivity={c: float(v) for c, v in zip(order, sensitivity, strict=True)},
        specificity={c: float(v) for c, v in zip(order, specificity, strict=True)},
        confusion=confusion,
        roc_auc=roc_auc,
        positive_label=positive if positive in order else None,
    )


def localization_errors(true_xy: np.ndarray, predicted_xy: np.ndarray) -> np.ndarray:
    """Euclidean error distance of every estimate.

    Examples
    --------
    >>> localization_errors(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])).tolist()
    [5.0]
    """
    truth = np.asarray(true_xy, dtype=np.float64)
    estimate = np.asarray(predicted_xy, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("positions must both be (n, 2)")
    return np.hypot(*(estimate - truth).T)


@dataclass(frozen=True)
class ErrorSummary:
    """Summary of localization error distances (metres)."""

    n_samples: int
    mean: float
    median: float
    p90: float
    rmse: float
    max: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """JSON-ready representation."""
        return asdict(self)


def summarize_errors(errors: np.ndarray) -> ErrorSummary:
    """Mean, median, 90th percentile, RMSE and maximum of error distances.

    Examples
    --------
    >>> s = summarize_errors(np.array([1.0, 2.0, 3.0, 4.0]))
    >>> s.mean, s.median, round(s.rmse, 4)
    (2.5, 2.5, 2.7386)
    """
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("errors must be a non-empty one-dimensional array")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("error distances must be finite and non-negative")
    return ErrorSummary(
        n_samples=int(values.size),
        mean=float(values.mean()),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        rmse=float(np.sqrt(np.mean(values**2))),
        max=float(values.max()),
    )
