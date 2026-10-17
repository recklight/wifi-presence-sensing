"""Estimating a usable noise floor from an unreliable per-packet noise field.

The noise value in a beamforming report cannot be taken at face value.  Some
reports carry the sentinel ``-127`` dBm ("not measured"), and in some captures
a large share of packets report a floor 20-30 dB above the thermal noise.
Used as-is, those values make every SNR-based quantity of the packet wrong
without any visible error.

:class:`NoiseModel` offers three strategies:

``reported``
    Use the field as-is (for comparison only).
``sentinel_guard``
    Replace the ``-127`` sentinel by a fixed fallback floor (-92 dBm).
``robust``
    Guard the sentinel, then apply a Hampel identifier (Hampel, 1974; Liu et
    al., 2004): values farther than ``threshold`` scaled MADs from the median
    are replaced by the median.  Finally clamp the result to a plausible
    range.  The default.

Every strategy returns the fraction of packets it replaced, and that number
goes into the run report; above :data:`REPLACED_FRACTION_WARNING` it is also
logged as a warning.  A capture where a third of the noise values had to be
overridden should be looked at, not just repaired.

References
----------
F. R. Hampel, "The Influence Curve and Its Role in Robust Estimation," Journal
of the American Statistical Association, 69(346):383-393, 1974.

H. Liu, S. Shah, W. Jiang, "On-line Outlier Detection and Data Cleaning,"
Computers and Chemical Engineering, 28(9):1635-1647, 2004.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_LOGGER = logging.getLogger(__name__)

#: Scale factor turning the MAD into a consistent estimate of a Gaussian sigma.
MAD_TO_SIGMA = 1.4826

#: Replacing more than this fraction of a trace's noise values is logged as a warning.
REPLACED_FRACTION_WARNING = 0.10


class NoiseStrategy(str, Enum):
    """How :class:`NoiseModel` treats the reported noise field."""

    REPORTED = "reported"
    SENTINEL_GUARD = "sentinel_guard"
    ROBUST = "robust"


def hampel_outliers(
    values: np.ndarray, *, threshold: float = 3.5, window: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Flag outliers with the Hampel identifier.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional series.
    threshold : float, optional
        Number of scaled MADs beyond which a value is an outlier.
    window : int, optional
        Odd window length for a rolling median/MAD.  ``None`` uses the whole
        series (a global median and MAD).

    Returns
    -------
    outliers : numpy.ndarray
        Boolean mask.
    reference : numpy.ndarray
        The (rolling) median each value was compared against.

    Examples
    --------
    >>> x = np.array([-92.0, -91.0, -93.0, -92.0, -60.0, -92.0])
    >>> hampel_outliers(x)[0].tolist()
    [False, False, False, False, True, False]
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {data.shape}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if data.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)

    if window is None:
        median = np.full(data.shape, np.median(data))
        mad = np.full(data.shape, np.median(np.abs(data - median[0])))
    else:
        if window < 3 or window % 2 == 0:
            raise ValueError(f"window must be an odd integer >= 3, got {window}")
        half = window // 2
        padded = np.pad(data, half, mode="edge")
        windows = sliding_window_view(padded, window)
        median = np.median(windows, axis=1)
        mad = np.median(np.abs(windows - median[:, None]), axis=1)

    sigma = MAD_TO_SIGMA * mad
    # A zero MAD (more than half the values identical) would flag every other
    # value; there, anything more than one unit (1 dB for noise values) from the
    # median counts as an outlier.
    sigma = np.where(sigma > 0, sigma, 1.0 / threshold)
    return np.abs(data - median) > threshold * sigma, median


@dataclass(frozen=True)
class NoiseEstimate:
    """Result of :meth:`NoiseModel.estimate`.

    Attributes
    ----------
    noise_dbm : numpy.ndarray
        Noise floor per packet after the strategy was applied.
    replaced : numpy.ndarray
        Boolean mask of packets whose value was overridden.
    strategy : str
        Strategy used.
    """

    noise_dbm: np.ndarray
    replaced: np.ndarray
    strategy: str

    @property
    def replaced_fraction(self) -> float:
        """Fraction of packets whose noise value was overridden."""
        return float(self.replaced.mean()) if self.replaced.size else 0.0


@dataclass(frozen=True)
class NoiseModel:
    """Turn the reported per-packet noise field into a usable noise floor.

    Parameters
    ----------
    strategy : str
        One of :class:`NoiseStrategy`.
    sentinel_dbm : float
        Value meaning "not measured".
    fallback_dbm : float
        Floor substituted for the sentinel (and used when nothing else is
        available).
    threshold : float
        Hampel threshold in scaled MADs (``robust``).
    window : int or None
        Rolling Hampel window; ``None`` for trace-wide statistics (``robust``).
    plausible_range_dbm : (float, float)
        Final clamp applied by ``robust``.

    Examples
    --------
    >>> model = NoiseModel()
    >>> est = model.estimate(np.array([-92, -91, -127, -93, -62, -92]))
    >>> est.noise_dbm.tolist(), round(est.replaced_fraction, 3)
    ([-92.0, -91.0, -92.0, -93.0, -92.0, -92.0], 0.333)
    """

    strategy: str = NoiseStrategy.ROBUST.value
    sentinel_dbm: float = -127.0
    fallback_dbm: float = -92.0
    threshold: float = 3.5
    window: int | None = None
    plausible_range_dbm: tuple[float, float] = (-105.0, -60.0)

    def __post_init__(self) -> None:
        try:
            NoiseStrategy(self.strategy)
        except ValueError:
            allowed = [s.value for s in NoiseStrategy]
            raise ValueError(f"noise strategy {self.strategy!r} is not one of {allowed}") from None
        low, high = self.plausible_range_dbm
        if not low < high:
            raise ValueError(
                f"plausible_range_dbm must be increasing, got {self.plausible_range_dbm}"
            )

    def estimate(self, reported_dbm: np.ndarray) -> NoiseEstimate:
        """Apply the strategy to a trace's reported noise values.

        Parameters
        ----------
        reported_dbm : numpy.ndarray
            Noise field of every packet, in arrival order.

        Returns
        -------
        NoiseEstimate
            Cleaned values and the replacement mask.
        """
        reported = np.asarray(reported_dbm, dtype=np.float64)
        strategy = NoiseStrategy(self.strategy)
        if strategy is NoiseStrategy.REPORTED:
            return NoiseEstimate(reported.copy(), np.zeros(reported.shape, bool), strategy.value)

        sentinel = reported == self.sentinel_dbm
        cleaned = np.where(sentinel, self.fallback_dbm, reported)
        replaced = sentinel.copy()
        if strategy is NoiseStrategy.ROBUST and cleaned.size:
            valid = ~sentinel
            if valid.any():
                outliers = np.zeros(cleaned.shape, dtype=bool)
                flagged, reference = hampel_outliers(
                    cleaned[valid], threshold=self.threshold, window=self.window
                )
                outliers[valid] = flagged
                substitute = np.full(cleaned.shape, self.fallback_dbm)
                substitute[valid] = reference
                cleaned = np.where(outliers, substitute, cleaned)
                replaced |= outliers
            low, high = self.plausible_range_dbm
            clamped = np.clip(cleaned, low, high)
            replaced |= clamped != cleaned
            cleaned = clamped

        estimate = NoiseEstimate(cleaned, replaced, strategy.value)
        level = (
            logging.WARNING
            if estimate.replaced_fraction > REPLACED_FRACTION_WARNING
            else logging.INFO
        )
        _LOGGER.log(
            level,
            "noise model %r replaced %.1f%% of the reported noise values",
            strategy.value,
            100.0 * estimate.replaced_fraction,
        )
        return estimate
