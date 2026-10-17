"""Cross-validation splits that respect temporal correlation.

Consecutive CSI windows from one recording are nearly copies of each other.  A
random train/test split puts neighbours on both sides, and the model is then
graded on data it has effectively seen: accuracy is inflated, sometimes to
near-perfect, while the model has learned the recording rather than the
phenomenon.

Every split here is *grouped*: all windows of a group (a recording, a session
or a contiguous time block) fall on the same side.  Classification folds are
additionally stratified so every fold sees every class.  The cost is that each
class needs at least ``n_splits`` groups; :func:`grouped_stratified_folds`
checks that up front and says how to fix it, instead of failing deep inside
scikit-learn.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

_LOGGER = logging.getLogger(__name__)


class SplitError(ValueError):
    """Raised when the data cannot support the requested grouped split."""


def groups_per_class(labels: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    """Number of distinct groups holding each class.

    Examples
    --------
    >>> groups_per_class(np.array(["a", "a", "b"]), np.array(["g1", "g2", "g1"]))
    {'a': 2, 'b': 1}
    """
    labels = np.asarray(labels).astype(str)
    groups = np.asarray(groups).astype(str)
    return {str(c): int(np.unique(groups[labels == c]).size) for c in np.unique(labels)}


def grouped_stratified_folds(
    labels: np.ndarray, groups: np.ndarray, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified group k-fold indices.

    Parameters
    ----------
    labels : numpy.ndarray
        Class of every sample.
    groups : numpy.ndarray
        Group of every sample; a group never spans train and test.
    n_splits : int
        Number of folds.
    seed : int
        Shuffling seed.

    Returns
    -------
    list of (train_indices, test_indices)
        The folds.

    Raises
    ------
    SplitError
        If labels and groups differ in length, fewer than two classes are
        present, or a class has fewer groups than folds.
    """
    labels = np.asarray(labels).astype(str)
    groups = np.asarray(groups).astype(str)
    if labels.shape != groups.shape:
        raise SplitError("labels and groups must have the same length")
    if n_splits < 2:
        raise SplitError("n_splits must be at least 2")
    counts = groups_per_class(labels, groups)
    if len(counts) < 2:
        raise SplitError(f"need at least two classes, found {list(counts)}")
    short = {c: n for c, n in counts.items() if n < n_splits}
    if short:
        raise SplitError(
            f"class(es) {short} have fewer groups than n_splits={n_splits}. Record more "
            "sessions, lower evaluation.n_splits, or set evaluation.group_by to 'block' "
            "to split long recordings into contiguous time blocks"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(labels.size), labels, groups))
    for train, test in folds:
        assert_disjoint_groups(train, test, groups)
    return folds


def grouped_folds(groups: np.ndarray, *, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group k-fold indices for regression targets (e.g. positions).

    ``n_splits`` is reduced to the number of groups when there are fewer.

    Raises
    ------
    SplitError
        If there are fewer than two groups.
    """
    groups = np.asarray(groups).astype(str)
    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        raise SplitError("need at least two groups for grouped cross-validation")
    splits = min(n_splits, n_groups)
    if splits < n_splits:
        _LOGGER.info("only %d group(s): using %d folds instead of %d", n_groups, splits, n_splits)
    folds = list(GroupKFold(n_splits=splits).split(np.zeros(groups.size), groups=groups))
    for train, test in folds:
        assert_disjoint_groups(train, test, groups)
    return folds


def assert_disjoint_groups(train: np.ndarray, test: np.ndarray, groups: np.ndarray) -> None:
    """Raise if any group appears on both sides of a split.

    Examples
    --------
    >>> assert_disjoint_groups(np.array([0, 1]), np.array([2]), np.array(["a", "a", "b"]))
    >>> try:
    ...     assert_disjoint_groups(np.array([0]), np.array([1]), np.array(["a", "a"]))
    ... except SplitError as error:
    ...     print(error)
    group(s) ['a'] appear in both train and test
    """
    groups = np.asarray(groups)
    shared = np.intersect1d(groups[train], groups[test])
    if shared.size:
        raise SplitError(f"group(s) {shared.astype(str).tolist()} appear in both train and test")


def iter_fold_groups(
    folds: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield the sorted train and test group names of every fold (for reports)."""
    groups = np.asarray(groups).astype(str)
    for train, test in folds:
        yield sorted(set(groups[train])), sorted(set(groups[test]))
