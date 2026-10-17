"""Fingerprint localization on a radio map, with three interchangeable feature views.

A radio map stores, for every reference position, the fingerprint observed
there: the per-feature mean (and spread) over the training windows recorded at
that position.  A new window is located by matching its fingerprint against
the map:

``knn``
    Deterministic nearest-neighbour matching in signal space (the RADAR
    approach, Bahl & Padmanabhan, 2000).  Every feature is measured in units
    of its pooled within-position spread, and the ``n_neighbors`` closest
    reference positions are averaged with weights
    :math:`\\exp(-\\tfrac12 (d_j^2 - d_{\\min}^2))`, so a neighbour only
    shares the estimate when its fingerprint fits nearly as well as the best
    one.  (Plain inverse-distance weights are almost uniform when there are
    hundreds of features, which drags every estimate towards the centroid of
    the neighbours.)
``probabilistic``
    Each reference position models every feature as an independent
    Gaussian; the estimate is the posterior-weighted mean of the
    ``n_neighbors`` most likely positions (in the spirit of Horus, Youssef &
    Agrawala, 2005).

The three locators share this algorithm and differ only in the evidence they
see, so a comparison between them isolates the value of the features:

=========================  ==============================  ==================
estimator                  feature groups                  method family
=========================  ==============================  ==================
``ReceivedPowerLocator``   ``power``                       RSS fingerprinting
``RawResponseLocator``     ``response``                    CSI amplitude
``ModalResponseLocator``   ``modal`` + ``profile``         EEMD modal features
=========================  ==============================  ==================

All three follow the scikit-learn estimator protocol (``fit(X, positions)``,
``predict(X)``).

The person being located carries no device: the radio map is surveyed by
standing at each reference position between a fixed transmitter and receiver,
as in device-free CSI fingerprinting (Xiao et al., 2013).  Using per-subcarrier
CSI amplitudes instead of a single received power as the fingerprint follows
FIFS (Xiao et al., 2012).

References
----------
P. Bahl, V. N. Padmanabhan, "RADAR: An In-Building RF-based User Location and
Tracking System," IEEE INFOCOM, 2000.

M. Youssef, A. Agrawala, "The Horus WLAN Location Determination System," ACM
MobiSys, 2005.

J. Xiao, K. Wu, Y. Yi, L. M. Ni, "FIFS: Fine-Grained Indoor Fingerprinting
System," IEEE ICCCN, 2012.

J. Xiao, K. Wu, Y. Yi, L. Wang, L. M. Ni, "Pilot: Passive Device-Free Indoor
Localization Using Channel State Information," IEEE ICDCS, 2013.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..features.extraction import GROUP_SEPARATOR

_LOGGER = logging.getLogger(__name__)

#: Matching methods accepted by :class:`RadioMapLocator`.
LOCATOR_METHODS: tuple[str, ...] = ("knn", "probabilistic")


class RadioMapLocator(RegressorMixin, BaseEstimator):
    """Radio-map fingerprint locator.

    Parameters
    ----------
    method : {"knn", "probabilistic"}
        Matching rule.
    n_neighbors : int
        Reference positions combined into each estimate.
    min_std : float
        Floor on per-feature standard deviations, in units of the pooled
        within-position spread.

    Examples
    --------
    >>> X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
    >>> xy = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [2, 0], [2, 0]], dtype=float)
    >>> locator = RadioMapLocator(n_neighbors=1).fit(X, xy)
    >>> locator.predict(np.array([[5.05], [9.9]])).tolist()
    [[1.0, 0.0], [2.0, 0.0]]
    """

    #: Feature groups the locator reads from a DataFrame (``None``: all columns).
    feature_groups: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, method: str = "knn", n_neighbors: int = 3, min_std: float = 1e-3) -> None:
        self.method = method
        self.n_neighbors = n_neighbors
        self.min_std = min_std

    def _select(self, X: pd.DataFrame | np.ndarray, *, fitting: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fitting:
                groups = type(self).feature_groups
                if groups is None:
                    columns = [str(c) for c in X.columns]
                else:
                    prefixes = tuple(f"{g}{GROUP_SEPARATOR}" for g in groups)
                    columns = [str(c) for c in X.columns if str(c).startswith(prefixes)]
                if not columns:
                    raise ValueError(
                        f"{type(self).__name__} needs feature group(s) {groups}, "
                        "which the table does not contain"
                    )
                self.selected_features_ = columns
            data = X[self.selected_features_].to_numpy(dtype=np.float64)
        else:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim != 2:
                raise ValueError("X must be two-dimensional")
            if fitting:
                self.selected_features_ = None
        if not np.all(np.isfinite(data)):
            raise ValueError("features contain NaN or infinite values")
        return data

    def fit(self, X: pd.DataFrame | np.ndarray, positions: np.ndarray) -> RadioMapLocator:
        """Build the radio map from training windows and their positions."""
        if self.method not in LOCATOR_METHODS:
            raise ValueError(f"method={self.method!r} is not one of {LOCATOR_METHODS}")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        data = self._select(X, fitting=True)
        xy = np.asarray(positions, dtype=np.float64)
        if xy.shape != (data.shape[0], 2) or not np.all(np.isfinite(xy)):
            raise ValueError("positions must be finite and shaped (n_samples, 2)")

        reference, inverse = np.unique(np.round(xy, 6), axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        counts = np.bincount(inverse, minlength=reference.shape[0]).astype(np.float64)
        sums = np.zeros((reference.shape[0], data.shape[1]))
        np.add.at(sums, inverse, data)
        means = sums / counts[:, None]
        residual = data - means[inverse]
        squares = np.zeros_like(sums)
        np.add.at(squares, inverse, residual**2)
        variances = squares / counts[:, None]

        # Scale every feature by its pooled within-position spread: a feature
        # that barely changes while the position is fixed but differs between
        # positions gets a large weight, a noisy one a small weight.
        repeated = counts > 1
        if repeated.any():
            within = np.sqrt((squares[repeated].sum(axis=0)) / (counts[repeated] - 1.0).sum())
        else:
            within = data.std(axis=0)
        fallback = data.std(axis=0)
        scale = np.where(within > 0, within, np.where(fallback > 0, fallback, 1.0))

        self.feature_mean_ = data.mean(axis=0)
        self.feature_scale_ = scale
        stds = np.sqrt(variances) / scale
        # Positions seen in a single window borrow the pooled spread (1 in these units).
        stds[~repeated] = 1.0
        self.reference_positions_ = reference
        self.reference_means_ = (means - self.feature_mean_) / scale
        self.reference_stds_ = np.maximum(stds, self.min_std)
        return self

    def _scores(self, standardized: np.ndarray) -> np.ndarray:
        """Matching score of every (query, reference) pair; higher is better."""
        means = self.reference_means_
        if self.method == "knn":
            distance_sq = (
                (standardized**2).sum(axis=1)[:, None]
                + (means**2).sum(axis=1)[None, :]
                - 2.0 * standardized @ means.T
            )
            return -0.5 * np.maximum(distance_sq, 0.0)
        inv_std = 1.0 / self.reference_stds_
        z = (standardized[:, None, :] - means[None, :, :]) * inv_std[None, :, :]
        return -0.5 * (z**2).sum(axis=2) + np.log(inv_std).sum(axis=1)[None, :]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Estimated ``(x, y)`` of every window."""
        data = self._select(X, fitting=False)
        standardized = (data - self.feature_mean_) / self.feature_scale_
        scores = self._scores(standardized)
        k = min(self.n_neighbors, scores.shape[1])
        nearest = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        chosen = np.take_along_axis(scores, nearest, axis=1)
        # Both scores are log-likelihood-like: knn's is -d^2/2 in units of the
        # within-position spread, the probabilistic one a Gaussian log density.
        weights = np.exp(chosen - chosen.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("qk,qkd->qd", weights, self.reference_positions_[nearest])


class ReceivedPowerLocator(RadioMapLocator):
    """Radio-map locator on received-power features (RSS fingerprinting baseline)."""

    feature_groups: ClassVar[tuple[str, ...] | None] = ("power",)


class RawResponseLocator(RadioMapLocator):
    """Radio-map locator on time-averaged CSI amplitudes (raw CSI baseline)."""

    feature_groups: ClassVar[tuple[str, ...] | None] = ("response",)


class ModalResponseLocator(RadioMapLocator):
    """Radio-map locator on EEMD modal and subcarrier-profile features (proposed)."""

    feature_groups: ClassVar[tuple[str, ...] | None] = ("modal", "profile")
