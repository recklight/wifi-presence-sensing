"""Empirical CDF of localization error distances.

Fingerprint locators are compared by the CDF of their error distances, as in
RADAR (Bahl & Padmanabhan, 2000).  A CDF value alone does not say how much
data it rests on, so :func:`error_distance_cdf` also keeps ``n_samples`` (the
denominator of every value), ``fraction_beyond_max`` (the share of errors past
the largest threshold, which is why a curve can stop below 1.0) and a
percentile-bootstrap confidence band (Efron, 1979) at every threshold.

The sorted error vector is kept together with its SHA-256 digest.  Locators
that read different feature groups should not end up with bit-identical error
vectors; if two do, the comparison is wired wrong (both reading the same
columns, for instance), and :func:`find_duplicate_curves` flags the pair.

References
----------
B. Efron, "Bootstrap Methods: Another Look at the Jackknife," The Annals of
Statistics, 7(1):1-26, 1979.

P. Bahl, V. N. Padmanabhan, "RADAR: An In-Building RF-based User Location and
Tracking System," IEEE INFOCOM, 2000.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import ErrorSummary, localization_errors, summarize_errors

_LOGGER = logging.getLogger(__name__)

#: Bootstrap replicates evaluated per chunk, to bound memory.
_BOOTSTRAP_CHUNK = 256


@dataclass(frozen=True, eq=False)
class ErrorCdfResult:
    """Error-distance CDF with its sample size, tail and confidence band.

    Attributes
    ----------
    thresholds_m : numpy.ndarray
        Distances at which the CDF is tabulated.
    cdf : numpy.ndarray
        Fraction of errors at or below each threshold.
    ci_low, ci_high : numpy.ndarray
        Bootstrap confidence band (NaN when ``n_bootstrap`` is 0).
    confidence : float
        Confidence level of the band.
    n_samples : int
        Number of error samples.
    fraction_beyond_max : float
        Fraction of errors above the largest threshold.
    errors : numpy.ndarray
        The raw error distances, sorted.
    summary : ErrorSummary
        Mean, median, 90th percentile, RMSE and maximum.
    digest : str
        SHA-256 of the sorted errors.
    """

    thresholds_m: np.ndarray
    cdf: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    confidence: float
    n_samples: int
    fraction_beyond_max: float
    errors: np.ndarray
    summary: ErrorSummary
    digest: str

    def as_dict(self) -> dict[str, object]:
        """JSON-ready representation (without the raw errors)."""
        return {
            "thresholds_m": self.thresholds_m.tolist(),
            "cdf": self.cdf.tolist(),
            "ci_low": self.ci_low.tolist(),
            "ci_high": self.ci_high.tolist(),
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "fraction_beyond_max": self.fraction_beyond_max,
            "summary": self.summary.as_dict(),
            "digest": self.digest,
        }

    def empirical_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """The full step CDF: sorted errors and cumulative probabilities."""
        n = self.errors.size
        return self.errors, np.arange(1, n + 1) / n


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def error_distance_cdf(
    true_xy: np.ndarray,
    predicted_xy: np.ndarray,
    *,
    thresholds_m: Sequence[float] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ErrorCdfResult:
    """Tabulate the error-distance CDF of position estimates.

    Parameters
    ----------
    true_xy, predicted_xy : numpy.ndarray
        ``(n, 2)`` positions in metres.
    thresholds_m : sequence of float, optional
        Increasing distances.
    n_bootstrap : int, optional
        Bootstrap resamples (0 disables the band).
    confidence : float, optional
        Confidence level of the band.
    seed : int, optional
        Bootstrap seed.

    Returns
    -------
    ErrorCdfResult
        CDF, band, sample size, tail fraction, raw errors and digest.

    Examples
    --------
    >>> truth = np.zeros((4, 2))
    >>> estimate = np.array([[0.5, 0], [1.5, 0], [2.5, 0], [9.0, 0]])
    >>> r = error_distance_cdf(truth, estimate, thresholds_m=[1, 2, 3], n_bootstrap=0)
    >>> r.cdf.tolist(), r.n_samples, r.fraction_beyond_max
    ([0.25, 0.5, 0.75], 4, 0.25)
    """
    errors = localization_errors(true_xy, predicted_xy)
    return error_cdf_from_errors(
        errors,
        thresholds_m=thresholds_m,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def error_cdf_from_errors(
    errors: np.ndarray,
    *,
    thresholds_m: Sequence[float] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ErrorCdfResult:
    """Same as :func:`error_distance_cdf`, from precomputed error distances."""
    values = np.sort(np.asarray(errors, dtype=np.float64))
    summary = summarize_errors(values)
    thresholds = np.asarray(thresholds_m, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be a non-empty increasing sequence")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")

    n = values.size
    cdf = np.searchsorted(values, thresholds, side="right") / n

    low = np.full(thresholds.size, np.nan)
    high = np.full(thresholds.size, np.nan)
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        replicates = np.empty((n_bootstrap, thresholds.size))
        for start in range(0, n_bootstrap, _BOOTSTRAP_CHUNK):
            count = min(_BOOTSTRAP_CHUNK, n_bootstrap - start)
            sample = values[rng.integers(0, n, size=(count, n))]
            replicates[start : start + count] = (sample[..., None] <= thresholds).mean(axis=1)
        tail = 100.0 * (1.0 - confidence) / 2.0
        low, high = np.percentile(replicates, [tail, 100.0 - tail], axis=0)

    return ErrorCdfResult(
        thresholds_m=thresholds,
        cdf=cdf,
        ci_low=low,
        ci_high=high,
        confidence=float(confidence),
        n_samples=int(n),
        fraction_beyond_max=float(np.mean(values > thresholds[-1])),
        errors=values,
        summary=summary,
        digest=_digest(values),
    )


def find_duplicate_curves(results: Mapping[str, ErrorCdfResult]) -> list[list[str]]:
    """Groups of result names whose error vectors are byte-for-byte identical.

    Examples
    --------
    >>> a = error_cdf_from_errors(np.array([1.0, 2.0]), n_bootstrap=0)
    >>> b = error_cdf_from_errors(np.array([2.0, 1.0]), n_bootstrap=0)
    >>> c = error_cdf_from_errors(np.array([1.0, 3.0]), n_bootstrap=0)
    >>> find_duplicate_curves({"x": a, "y": b, "z": c})
    [['x', 'y']]
    """
    by_digest: dict[str, list[str]] = {}
    for name, result in results.items():
        by_digest.setdefault(result.digest, []).append(name)
    duplicates = [sorted(names) for names in by_digest.values() if len(names) > 1]
    for names in duplicates:
        _LOGGER.warning(
            "error curves %s are bit-identical; the locators probably read the same "
            "feature columns",
            names,
        )
    return duplicates
