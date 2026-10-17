"""Grouped cross-validated comparison of the three fingerprint locators.

Every locator is fitted and evaluated on the same grouped folds (a session
never contributes to both the radio map and the test windows), and each one
gets its own error vector, CDF, confidence band and digest.  Bit-identical
curves are flagged (see
:func:`~wifi_presence_sensing.evaluate.error_cdf.find_duplicate_curves`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from ..evaluate.error_cdf import ErrorCdfResult, error_distance_cdf, find_duplicate_curves
from ..evaluate.splitting import grouped_folds
from ..features.extraction import FeatureTable
from .fingerprint import (
    ModalResponseLocator,
    RadioMapLocator,
    RawResponseLocator,
    ReceivedPowerLocator,
)

_LOGGER = logging.getLogger(__name__)


def default_locators(method: str = "knn", n_neighbors: int = 3) -> dict[str, RadioMapLocator]:
    """The three locators compared by :func:`benchmark_locators`."""
    return {
        "received_power": ReceivedPowerLocator(method=method, n_neighbors=n_neighbors),
        "raw_response": RawResponseLocator(method=method, n_neighbors=n_neighbors),
        "modal_response": ModalResponseLocator(method=method, n_neighbors=n_neighbors),
    }


@dataclass(eq=False)
class LocalizationBenchmark:
    """Results of :func:`benchmark_locators`.

    Attributes
    ----------
    results : dict
        ``{locator name: ErrorCdfResult}``.
    predictions : dict
        ``{locator name: out-of-fold (x, y) estimates}``.
    duplicates : list of list of str
        Locators whose error vectors are identical.
    n_folds : int
        Number of grouped folds.
    """

    results: dict[str, ErrorCdfResult]
    predictions: dict[str, np.ndarray]
    duplicates: list[list[str]]
    n_folds: int

    def as_dict(self) -> dict[str, object]:
        """JSON-ready representation."""
        return {
            "n_folds": self.n_folds,
            "duplicates": self.duplicates,
            "results": {name: result.as_dict() for name, result in self.results.items()},
        }


def benchmark_locators(
    table: FeatureTable,
    *,
    locators: Mapping[str, RadioMapLocator] | None = None,
    group_by: str = "session",
    n_splits: int = 5,
    thresholds_m: Sequence[float] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> LocalizationBenchmark:
    """Cross-validate several locators and tabulate their error CDFs.

    Parameters
    ----------
    table : FeatureTable
        Windows with ground-truth positions.
    locators : mapping, optional
        ``{name: locator}``; defaults to :func:`default_locators`.  Locators
        whose feature groups are absent from the table are skipped.
    group_by : str, optional
        Cross-validation grouping.
    n_splits : int, optional
        Maximum number of folds.
    thresholds_m, n_bootstrap, confidence, seed
        Passed to :func:`~wifi_presence_sensing.evaluate.error_cdf.error_distance_cdf`.

    Returns
    -------
    LocalizationBenchmark
        Error CDFs and out-of-fold estimates.

    Raises
    ------
    ValueError
        If the table lacks positions or no locator is usable.
    """
    if not table.has_positions:
        raise ValueError(
            "localization needs a ground-truth position for every window; add 'position' "
            "to each recording of the dataset manifest"
        )
    candidates = dict(locators) if locators is not None else default_locators()
    frame = table.feature_frame()
    usable: dict[str, RadioMapLocator] = {}
    for name, locator in candidates.items():
        groups = type(locator).feature_groups
        if groups is not None and not table.column_indices(groups):
            _LOGGER.warning("%s skipped: the table has no %s features", name, groups)
            continue
        usable[name] = locator
    if not usable:
        raise ValueError("no locator can run on this feature table")

    folds = grouped_folds(table.groups(group_by), n_splits=n_splits)
    predictions = {name: np.full((table.n_windows, 2), np.nan) for name in usable}
    for train, test in folds:
        for name, locator in usable.items():
            model = clone(locator).fit(frame.iloc[train], table.positions[train])
            predictions[name][test] = model.predict(frame.iloc[test])

    results = {
        name: error_distance_cdf(
            table.positions,
            estimate,
            thresholds_m=thresholds_m,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        for name, estimate in predictions.items()
    }
    for name, result in results.items():
        _LOGGER.info(
            "%s: median error %.2f m, 90th percentile %.2f m (n=%d)",
            name,
            result.summary.median,
            result.summary.p90,
            result.n_samples,
        )
    return LocalizationBenchmark(
        results=results,
        predictions=predictions,
        duplicates=find_duplicate_curves(results),
        n_folds=len(folds),
    )
