"""Features of the amplitude profile across subcarriers.

Multipath makes the channel frequency-selective: some subcarriers sit in deep
fades, others on peaks.  A moving body perturbs the paths, which changes both
how much each subcarrier fluctuates over time and how coherently neighbouring
subcarriers fluctuate together.  These features summarize one window of one
antenna pair, shaped ``(n_time, n_subcarriers)``, in dB:

``temporal_std_mean`` / ``temporal_std_max``
    Mean and maximum over subcarriers of the temporal standard deviation.
``frequency_selectivity_db``
    Standard deviation across subcarriers of the time-averaged amplitude.
``spectral_range_db``
    Peak-to-peak of the time-averaged amplitude across subcarriers.
``subcarrier_correlation``
    Mean pairwise correlation of the subcarriers' temporal fluctuations.

The module also reduces the 30 subcarriers of an antenna pair to one series for
decomposition: the first principal component of the fluctuations (which pools
the coherent motion signal and averages out per-subcarrier noise) or the plain
subcarrier mean.
"""

from __future__ import annotations

import numpy as np

#: Profile metrics, in feature order.
PROFILE_METRICS: tuple[str, ...] = (
    "temporal_std_mean",
    "temporal_std_max",
    "frequency_selectivity_db",
    "spectral_range_db",
    "subcarrier_correlation",
)

#: Accepted values of the ``method`` argument of :func:`reduce_subcarriers`.
REDUCTION_METHODS: tuple[str, ...] = ("pca", "mean")


def _check_window(window: np.ndarray) -> np.ndarray:
    data = np.asarray(window, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError(f"window must be shaped (n_time >= 2, n_subcarriers), got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("window contains NaN or infinite values")
    return data


def subcarrier_profile(window_db: np.ndarray) -> dict[str, float]:
    """Profile metrics of one antenna pair over one window.

    Parameters
    ----------
    window_db : numpy.ndarray
        Amplitude in dB, ``(n_time, n_subcarriers)``.

    Returns
    -------
    dict
        One float per entry of :data:`PROFILE_METRICS`.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> common = rng.normal(size=(100, 1))
    >>> metrics = subcarrier_profile(common + 0.01 * rng.normal(size=(100, 30)))
    >>> metrics["subcarrier_correlation"] > 0.99
    True
    """
    data = _check_window(window_db)
    temporal_std = data.std(axis=0)
    profile = data.mean(axis=0)
    fluctuations = data - profile
    norms = np.sqrt((fluctuations**2).sum(axis=0))
    active = norms > 0
    if active.sum() >= 2:
        unit = fluctuations[:, active] / norms[active]
        correlation = unit.T @ unit
        n = correlation.shape[0]
        mean_correlation = float((correlation.sum() - np.trace(correlation)) / (n * (n - 1)))
    else:
        mean_correlation = 0.0
    return {
        "temporal_std_mean": float(temporal_std.mean()),
        "temporal_std_max": float(temporal_std.max()),
        "frequency_selectivity_db": float(profile.std()),
        "spectral_range_db": float(np.ptp(profile)),
        "subcarrier_correlation": mean_correlation,
    }


def reduce_subcarriers(window_db: np.ndarray, method: str = "pca") -> np.ndarray:
    """Collapse the subcarriers of one antenna pair to a single series.

    Parameters
    ----------
    window_db : numpy.ndarray
        Amplitude in dB, ``(n_time, n_subcarriers)``.
    method : {"pca", "mean"}, optional
        ``pca``: projection on the first principal component of the
        fluctuations, sign-aligned with the subcarrier mean and offset by the
        mean level.  ``mean``: average over subcarriers.

    Returns
    -------
    numpy.ndarray
        ``(n_time,)``.

    Examples
    --------
    >>> t = np.linspace(0, 1, 50)
    >>> window = np.outer(np.sin(2 * np.pi * 3 * t), np.linspace(1, 2, 30)) + 10.0
    >>> series = reduce_subcarriers(window, "pca")
    >>> bool(np.corrcoef(series, np.sin(2 * np.pi * 3 * t))[0, 1] > 0.999)
    True
    """
    data = _check_window(window_db)
    if method == "mean":
        return data.mean(axis=1)
    if method != "pca":
        raise ValueError(f"method={method!r} is not one of {REDUCTION_METHODS}")
    level = data.mean()
    centred = data - data.mean(axis=0)
    if not np.any(centred):
        return np.full(data.shape[0], level)
    left, singular, _ = np.linalg.svd(centred, full_matrices=False)
    component = left[:, 0] * singular[0] / np.sqrt(data.shape[1])
    reference = centred.mean(axis=1)
    if float(component @ reference) < 0:
        component = -component
    return component + level


def profile_feature_vector(
    window_db: np.ndarray, stream_names: list[str], *, prefix: str = "profile"
) -> tuple[np.ndarray, list[str]]:
    """Profile metrics of every antenna pair of a window.

    Parameters
    ----------
    window_db : numpy.ndarray
        ``(n_time, n_streams, n_subcarriers)`` in dB.
    stream_names : list of str
        One name per antenna pair (e.g. ``"tx0rx1"``).
    prefix : str, optional
        Feature-name prefix.

    Returns
    -------
    values : numpy.ndarray
        ``n_streams * len(PROFILE_METRICS)`` values.
    names : list of str
        ``"{prefix}__{stream}_{metric}"``.
    """
    data = np.asarray(window_db, dtype=np.float64)
    if data.ndim != 3 or data.shape[1] != len(stream_names):
        raise ValueError("window must be (n_time, n_streams, n_subcarriers) with named streams")
    values: list[float] = []
    names: list[str] = []
    for index, stream in enumerate(stream_names):
        metrics = subcarrier_profile(data[:, index, :])
        values.extend(metrics[name] for name in PROFILE_METRICS)
        names.extend(f"{prefix}__{stream}_{name}" for name in PROFILE_METRICS)
    return np.asarray(values), names
