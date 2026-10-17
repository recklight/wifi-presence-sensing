"""Features of the intrinsic mode functions of a CSI amplitude series.

A person moving through a room changes the multipath geometry.  In the CSI
amplitude this shows up as extra variation at the time scales of the motion,
not as a different average level: the mean amplitude of an occupied and an
empty room can differ by less than a decibel.  The IMFs separate that
variation by time scale, and the statistics below describe each scale:

``variance``
    Power of the mode.
``energy_fraction``
    Share of the total oscillatory energy (IMFs plus the detrended residual)
    held by the mode.
``mean_frequency_hz``
    Mean instantaneous frequency weighted by the instantaneous power
    :math:`a_k^2(t)`, from the analytic signal
    :math:`c_k + i\\,\\mathcal{H}[c_k] = a_k e^{i\\theta_k}` (Huang et al., 1998).
``zero_crossing_rate_hz``
    Zero crossings per second divided by two: a frequency estimate that does
    not depend on the Hilbert transform.

Every series gets the same number of modes (missing ones contribute zeros),
so the feature vector has a fixed layout whatever the decomposition returned.

References
----------
N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum
for nonlinear and non-stationary time series analysis," Proceedings of the
Royal Society of London A, 454(1971):903-995, 1998.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ..decompose.sifting import count_zero_crossings

#: Statistics computed for every mode, in feature order.
MODAL_STATISTICS: tuple[str, ...] = (
    "variance",
    "energy_fraction",
    "mean_frequency_hz",
    "zero_crossing_rate_hz",
)


def instantaneous_frequency(
    mode: np.ndarray, sampling_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequency and amplitude of IMFs via the Hilbert transform.

    Parameters
    ----------
    mode : numpy.ndarray
        One IMF, or several stacked along the leading axes; time runs along
        the last axis.  All modes are transformed in one call.
    sampling_rate_hz : float
        Sampling rate.

    Returns
    -------
    frequency_hz : numpy.ndarray
        One sample shorter than ``mode`` along the time axis (the midpoints
        between samples); clipped to ``[0, sampling_rate_hz / 2]``.
    amplitude : numpy.ndarray
        Instantaneous amplitude at the same midpoints.

    Examples
    --------
    >>> fs = 50.0
    >>> t = np.arange(500) / fs
    >>> freq, _ = instantaneous_frequency(np.cos(2 * np.pi * 3.0 * t), fs)
    >>> round(float(np.median(freq)), 3)
    3.0
    >>> tones = np.stack([np.cos(2 * np.pi * f * t) for f in (3.0, 7.0)])
    >>> np.round(np.median(instantaneous_frequency(tones, fs)[0], axis=1), 3).tolist()
    [3.0, 7.0]
    """
    if sampling_rate_hz <= 0:
        raise ValueError(f"sampling rate must be positive, got {sampling_rate_hz}")
    analytic = hilbert(np.asarray(mode, dtype=np.float64), axis=-1)
    phase = np.unwrap(np.angle(analytic), axis=-1)
    frequency = np.diff(phase, axis=-1) * sampling_rate_hz / (2.0 * np.pi)
    envelope = np.abs(analytic)
    amplitude = 0.5 * (envelope[..., 1:] + envelope[..., :-1])
    return np.clip(frequency, 0.0, sampling_rate_hz / 2.0), amplitude


def mode_statistics(
    imfs: np.ndarray, residual: np.ndarray, sampling_rate_hz: float
) -> dict[str, np.ndarray]:
    """Per-mode statistics of a decomposition.

    Parameters
    ----------
    imfs : numpy.ndarray
        ``(n_modes, n_samples)``.
    residual : numpy.ndarray
        ``(n_samples,)``.
    sampling_rate_hz : float
        Sampling rate.

    Returns
    -------
    dict
        One array of length ``n_modes`` per entry of :data:`MODAL_STATISTICS`.
    """
    modes = np.atleast_2d(np.asarray(imfs, dtype=np.float64))
    if modes.size == 0:
        return {name: np.zeros(0) for name in MODAL_STATISTICS}
    n_samples = modes.shape[1]
    duration = n_samples / sampling_rate_hz
    centred = modes - modes.mean(axis=1, keepdims=True)
    variance = centred.var(axis=1)
    mode_energy = (modes**2).sum(axis=1)
    detrended = np.asarray(residual, dtype=np.float64)
    detrended = detrended - detrended.mean()
    total = mode_energy.sum() + float((detrended**2).sum())
    energy_fraction = mode_energy / total if total > 0 else np.zeros_like(mode_energy)

    # Power-weighted (amplitude squared) mean instantaneous frequency of every
    # mode at once; an all-zero (padded) mode has zero weight everywhere and
    # gets frequency 0.
    frequency, amplitude = instantaneous_frequency(modes, sampling_rate_hz)
    weight = amplitude**2
    total_weight = weight.sum(axis=1)
    weighted = (frequency * weight).sum(axis=1)
    mean_frequency = np.divide(
        weighted, total_weight, out=np.zeros_like(weighted), where=total_weight > 0
    )

    crossings = np.array([count_zero_crossings(mode) for mode in modes], dtype=np.float64)
    return {
        "variance": variance,
        "energy_fraction": energy_fraction,
        "mean_frequency_hz": mean_frequency,
        "zero_crossing_rate_hz": crossings / (2.0 * duration),
    }


def modal_feature_vector(
    imfs: np.ndarray,
    residual: np.ndarray,
    sampling_rate_hz: float,
    *,
    n_modes: int,
    prefix: str = "modal",
) -> tuple[np.ndarray, list[str]]:
    """Fixed-length feature vector from a decomposition.

    Parameters
    ----------
    imfs : numpy.ndarray
        ``(n_found, n_samples)``; truncated or zero-padded to ``n_modes``.
    residual : numpy.ndarray
        The residual trend.
    sampling_rate_hz : float
        Sampling rate.
    n_modes : int
        Number of modes in the layout.
    prefix : str, optional
        Feature-name prefix.

    Returns
    -------
    values : numpy.ndarray
        ``n_modes * len(MODAL_STATISTICS) + 1`` values; the last is the slope
        of the residual trend per second.
    names : list of str
        ``"{prefix}_m{k}_{statistic}"`` and ``"{prefix}_trend_slope"``.

    Examples
    --------
    >>> fs = 20.0
    >>> t = np.arange(200) / fs
    >>> values, names = modal_feature_vector(np.sin(2 * np.pi * 2 * t)[None], 0 * t, fs, n_modes=2)
    >>> len(values), names[0], names[-1]
    (9, 'modal_m1_variance', 'modal_trend_slope')
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be at least 1, got {n_modes}")
    modes = np.atleast_2d(np.asarray(imfs, dtype=np.float64))
    n_samples = np.asarray(residual).size
    layout = np.zeros((n_modes, n_samples))
    taken = min(n_modes, modes.shape[0]) if modes.size else 0
    layout[:taken] = modes[:taken]
    stats = mode_statistics(layout, residual, sampling_rate_hz)

    values = [stats[name][k] for k in range(n_modes) for name in MODAL_STATISTICS]
    names = [f"{prefix}_m{k + 1}_{name}" for k in range(n_modes) for name in MODAL_STATISTICS]
    time = np.arange(n_samples) / sampling_rate_hz
    slope = float(np.polyfit(time, residual, 1)[0]) if n_samples >= 2 else 0.0
    values.append(slope)
    names.append(f"{prefix}_trend_slope")
    return np.asarray(values, dtype=np.float64), names
