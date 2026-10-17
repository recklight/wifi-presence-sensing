"""Grouped, stratified cross-validation of classifiers with baselines side by side.

:func:`cross_validate_grouped` runs one estimator through
:func:`~wifi_presence_sensing.evaluate.splitting.grouped_stratified_folds`, collects the
out-of-fold predictions and probabilities, and reports both per-fold metrics
(mean and spread) and metrics of the pooled out-of-fold predictions.  With a
parameter grid it tunes inside each training fold with an inner grouped split
(nested cross-validation), so the reported score never benefits from the
tuning (Varma & Simon, 2006).

:func:`compare_classifiers` evaluates several candidates on the *same* folds,
so differences between them are not differences between random splits.

References
----------
S. Varma, R. Simon, "Bias in Error Estimation When Using Cross-Validation for
Model Selection," BMC Bioinformatics, 7:91, 2006.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .metrics import ClassificationMetrics, classification_metrics
from .splitting import grouped_stratified_folds, groups_per_class, iter_fold_groups

_LOGGER = logging.getLogger(__name__)


@dataclass(eq=False)
class CrossValidationReport:
    """Outcome of a grouped cross-validation.

    Attributes
    ----------
    name : str
        Candidate name.
    pooled : ClassificationMetrics
        Metrics of all out-of-fold predictions together.
    fold_metrics : list of ClassificationMetrics
        Metrics of every test fold.
    fold_groups : list of dict
        Train and test groups of every fold.
    predictions : numpy.ndarray
        Out-of-fold predicted labels.
    probabilities : numpy.ndarray or None
        Out-of-fold class probabilities in ``pooled.labels`` order.
    best_params : list of dict
        Parameters chosen by the inner search in every fold (nested CV).
    """

    name: str
    pooled: ClassificationMetrics
    fold_metrics: list[ClassificationMetrics]
    fold_groups: list[dict[str, list[str]]]
    predictions: np.ndarray
    probabilities: np.ndarray | None
    best_params: list[dict[str, Any]] = field(default_factory=list)

    def fold_summary(self) -> dict[str, dict[str, float]]:
        """Mean and standard deviation over folds of each headline metric."""
        keys = self.fold_metrics[0].headline().keys() if self.fold_metrics else []
        summary: dict[str, dict[str, float]] = {}
        for key in keys:
            values = np.array(
                [m.headline()[key] for m in self.fold_metrics if m.headline()[key] is not None],
                dtype=np.float64,
            )
            if values.size:
                summary[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
        return summary

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "name": self.name,
            "pooled": self.pooled.as_dict(),
            "folds": self.fold_summary(),
            "fold_groups": self.fold_groups,
            "best_params": self.best_params,
        }


def _aligned_probabilities(model: Any, X: pd.DataFrame, labels: Sequence[str]) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    aligned = np.zeros((raw.shape[0], len(labels)))
    for column, cls in enumerate(np.asarray(model.classes_).astype(str)):
        aligned[:, list(labels).index(cls)] = raw[:, column]
    return aligned


def cross_validate_grouped(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
    positive_label: str | None = None,
    name: str = "model",
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    inner_splits: int = 3,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> CrossValidationReport:
    """Cross-validate one estimator with grouped, stratified folds.

    Parameters
    ----------
    estimator : scikit-learn classifier
        Cloned for every fold.
    X : pandas.DataFrame
        Features (all groups; the estimator selects what it uses).
    y : numpy.ndarray
        Labels.
    groups : numpy.ndarray
        Cross-validation group of every row.
    n_splits : int, optional
        Outer folds.
    seed : int, optional
        Fold shuffling seed.
    positive_label : str, optional
        Class treated as positive in the headline metrics.
    name : str, optional
        Name stored in the report.
    param_grid : mapping, optional
        Enables nested CV: the grid is searched on each training fold with
        an inner grouped split, scored by balanced accuracy (UAR).
    inner_splits : int, optional
        Folds of the inner search.
    folds : list, optional
        Precomputed folds (used by :func:`compare_classifiers`).

    Returns
    -------
    CrossValidationReport
        Pooled and per-fold metrics.
    """
    labels_y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)
    if len(X) != labels_y.size or groups.size != labels_y.size:
        raise ValueError("X, y and groups must have the same number of rows")
    if folds is None:
        folds = grouped_stratified_folds(labels_y, groups, n_splits=n_splits, seed=seed)
    labels = sorted(np.unique(labels_y).tolist())

    predictions = np.empty(labels_y.size, dtype=object)
    probabilities = np.zeros((labels_y.size, len(labels)))
    have_probabilities = True
    fold_metrics: list[ClassificationMetrics] = []
    best_params: list[dict[str, Any]] = []

    for index, (train, test) in enumerate(folds):
        model = clone(estimator)
        X_train, X_test = X.iloc[train], X.iloc[test]
        if param_grid:
            inner_counts = groups_per_class(labels_y[train], groups[train])
            splits = min(inner_splits, min(inner_counts.values()))
            if splits >= 2:
                search = GridSearchCV(
                    model,
                    dict(param_grid),
                    cv=StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed),
                    scoring="balanced_accuracy",
                    refit=True,
                )
                search.fit(X_train, labels_y[train], groups=groups[train])
                model = search.best_estimator_
                best_params.append(dict(search.best_params_))
            else:
                _LOGGER.warning(
                    "fold %d: too few groups for an inner search; using default parameters", index
                )
                model.fit(X_train, labels_y[train])
                best_params.append({})
        else:
            model.fit(X_train, labels_y[train])

        fold_prediction = np.asarray(model.predict(X_test)).astype(str)
        predictions[test] = fold_prediction
        fold_proba = _aligned_probabilities(model, X_test, labels)
        if fold_proba is None:
            have_probabilities = False
        else:
            probabilities[test] = fold_proba
        fold_metrics.append(
            classification_metrics(
                labels_y[test],
                fold_prediction,
                labels=labels,
                scores=fold_proba,
                positive_label=positive_label,
            )
        )

    pooled = classification_metrics(
        labels_y,
        predictions.astype(str),
        labels=labels,
        scores=probabilities if have_probabilities else None,
        positive_label=positive_label,
    )
    fold_groups = [
        {"train": train_groups, "test": test_groups}
        for train_groups, test_groups in iter_fold_groups(folds, groups)
    ]
    _LOGGER.info(
        "%s: UAR %.3f, accuracy %.3f over %d grouped folds",
        name,
        pooled.uar,
        pooled.accuracy,
        len(folds),
    )
    return CrossValidationReport(
        name=name,
        pooled=pooled,
        fold_metrics=fold_metrics,
        fold_groups=fold_groups,
        predictions=predictions.astype(str),
        probabilities=probabilities if have_probabilities else None,
        best_params=best_params,
    )


def compare_classifiers(
    candidates: Mapping[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
    positive_label: str | None = None,
    param_grids: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    inner_splits: int = 3,
) -> dict[str, CrossValidationReport]:
    """Evaluate several classifiers on identical grouped folds.

    Parameters
    ----------
    candidates : mapping
        ``{name: estimator}``.
    X, y, groups
        As for :func:`cross_validate_grouped`.
    n_splits, seed, positive_label, inner_splits
        As for :func:`cross_validate_grouped`.
    param_grids : mapping, optional
        ``{name: grid}`` for candidates that should be tuned (nested CV).

    Returns
    -------
    dict
        ``{name: CrossValidationReport}``.
    """
    labels_y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)
    folds = grouped_stratified_folds(labels_y, groups, n_splits=n_splits, seed=seed)
    grids = param_grids or {}
    return {
        name: cross_validate_grouped(
            estimator,
            X,
            labels_y,
            groups,
            seed=seed,
            positive_label=positive_label,
            name=name,
            param_grid=grids.get(name),
            inner_splits=inner_splits,
            folds=folds,
        )
        for name, estimator in candidates.items()
    }
