"""Timestamps: counter unwrapping, gap detection and sampling diagnostics.

The NIC timestamp is a 32-bit microsecond counter that wraps every ~71.6
minutes.  A capture that crosses the wrap appears to jump back by ~4295
seconds; left alone, the trace gets a negative duration and a meaningless
packet rate.  :func:`unwrap_timestamps` removes the wrap.

Captures are also rarely uniformly sampled: packets arrive in bursts separated
by pauses, and the pauses can be longer than any motion of interest.
:func:`sampling_diagnostics` measures what a trace can actually resolve, and
:func:`contiguous_segments` splits a trace where the pauses are too long to
interpolate across.

The motion bands behind that check follow the CSI sensing literature:
breathing modulates CSI at roughly 0.2-0.5 Hz (Liu et al., 2014), while a
reflector moving at speed ``v`` shifts a path length at ``2 v / wavelength``
cycles per second, i.e. several to tens of hertz for walking at 2.4/5 GHz
(Wang et al., 2015).

References
----------
X. Liu, J. Cao, S. Tang, J. Wen, "Wi-Sleep: Contactless Sleep Monitoring via
WiFi Signals," IEEE Real-Time Systems Symposium (RTSS), 2014.

W. Wang, A. X. Liu, M. Shahzad, K. Ling, S. Lu, "Understanding and Modeling of
WiFi Signal Based Human Activity Recognition," ACM MobiCom, 2015.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from .snapshot import TIMESTAMP_PERIOD_US

_LOGGER = logging.getLogger(__name__)

#: Intervals shorter than this are counted as part of a burst.
BURST_INTERVAL_S = 0.010

#: A forward step within this many microseconds of a full counter period is a
#: late packet across the wrap.  The window is kept narrow on purpose: with half
#: a period, a pause of 35.8 to 71.6 minutes would be folded into a negative step.
REORDER_TOLERANCE_US = 1_000_000

#: Representative motion bands used to state what a sampling rate can resolve.
#: Breathing sits around 0.2-0.5 Hz; walking produces Doppler shifts of
#: several hertz to tens of hertz at 2.4/5 GHz.
MOTION_BANDS_HZ: dict[str, tuple[float, float]] = {
    "breathing": (0.2, 0.5),
    "limb and torso motion": (0.5, 5.0),
    "walking Doppler": (5.0, 40.0),
}


def unwrap_timestamps(timestamp_us: np.ndarray) -> np.ndarray:
    """Remove the 32-bit wrap-around from a sequence of NIC timestamps.

    A step between consecutive packets that goes back by more than half the
    counter period is a wrap, and a period is added to it.  A step that goes
    forward by more than a period minus :data:`REORDER_TOLERANCE_US` is a
    packet stamped just before the wrap that turns up after it, so a period
    is subtracted instead of counting a second wrap.  Every other step is
    kept: a long pause stays a forward step, and the backward steps that
    remain are out-of-order packets, which stay in place and are logged.

    One case cannot be settled from the counter alone: a pause longer than
    half a period with a wrap inside it looks like reordering and comes out
    as a backward step.

    Parameters
    ----------
    timestamp_us : numpy.ndarray
        Raw counter values in microseconds, in arrival order.

    Returns
    -------
    numpy.ndarray
        ``int64`` microseconds, monotone across wraps.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or holds values outside the
        counter range.

    Examples
    --------
    >>> raw = np.array([4_294_967_000, 4_294_967_200, 104, 304])
    >>> unwrap_timestamps(raw).tolist()
    [4294967000, 4294967200, 4294967400, 4294967600]
    """
    values = np.asarray(timestamp_us, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"timestamps must be one-dimensional, got shape {values.shape}")
    if values.size and (values.min() < 0 or values.max() >= TIMESTAMP_PERIOD_US):
        raise ValueError("timestamps must lie in the 32-bit counter range [0, 2**32)")
    if values.size < 2:
        return values.copy()

    period = TIMESTAMP_PERIOD_US
    steps = np.diff(values)
    steps = np.where(steps < -(period // 2), steps + period, steps)
    steps = np.where(steps > period - REORDER_TOLERANCE_US, steps - period, steps)
    reordered = np.count_nonzero(steps < 0)
    if reordered:
        _LOGGER.warning(
            "%d backwards timestamp step(s) smaller than a counter wrap: packets out of order",
            reordered,
        )
    return values[0] + np.concatenate([[0], np.cumsum(steps, dtype=np.int64)])


def find_gaps(times_s: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Indices ``i`` where the interval ``times_s[i+1] - times_s[i]`` exceeds ``max_gap_s``.

    Examples
    --------
    >>> find_gaps(np.array([0.0, 0.1, 0.2, 3.0, 3.1]), 1.0).tolist()
    [2]
    """
    times = np.asarray(times_s, dtype=np.float64)
    if max_gap_s <= 0:
        raise ValueError(f"max_gap_s must be positive, got {max_gap_s}")
    return np.flatnonzero(np.diff(times) > max_gap_s)


def contiguous_segments(
    times_s: np.ndarray, max_gap_s: float, *, min_samples: int = 2
) -> list[slice]:
    """Split a timeline at every gap longer than ``max_gap_s``.

    Parameters
    ----------
    times_s : numpy.ndarray
        Monotone sample times in seconds.
    max_gap_s : float
        Largest interval that is still bridged by interpolation.
    min_samples : int, optional
        Segments with fewer samples are dropped.

    Returns
    -------
    list of slice
        Index ranges of the retained segments.

    Examples
    --------
    >>> contiguous_segments(np.array([0.0, 0.1, 0.2, 3.0, 3.1]), 1.0)
    [slice(0, 3, None), slice(3, 5, None)]
    """
    times = np.asarray(times_s, dtype=np.float64)
    if times.size == 0:
        return []
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be sorted; unwrap and sort them first")
    cuts = find_gaps(times, max_gap_s) + 1
    bounds = np.concatenate([[0], cuts, [times.size]])
    return [
        slice(int(start), int(stop))
        for start, stop in pairwise(bounds)
        if stop - start >= min_samples
    ]


@dataclass(frozen=True)
class SamplingDiagnostics:
    """What a timeline can and cannot resolve.

    Attributes
    ----------
    n_samples : int
        Number of packets.
    duration_s : float
        First-to-last span.
    median_interval_s : float
        Median packet interval.
    median_rate_hz : float
        ``1 / median_interval_s``: the rate inside bursts.
    mean_rate_hz : float
        ``(n_samples - 1) / duration_s``: the long-run rate including pauses.
    nyquist_hz : float
        Half the mean rate: the highest frequency a uniform resampling at the
        long-run rate can represent.
    burst_fraction : float
        Fraction of intervals shorter than :data:`BURST_INTERVAL_S`.
    interval_p95_s : float
        95th-percentile interval.
    max_interval_s : float
        Longest pause.
    n_gaps : int
        Pauses longer than the configured maximum gap.
    interval_cv : float
        Coefficient of variation of the intervals; 0 for perfectly regular
        sampling.
    """

    n_samples: int
    duration_s: float
    median_interval_s: float
    median_rate_hz: float
    mean_rate_hz: float
    nyquist_hz: float
    burst_fraction: float
    interval_p95_s: float
    max_interval_s: float
    n_gaps: int
    interval_cv: float

    def as_dict(self) -> dict[str, float]:
        """Return the diagnostics as a plain dictionary."""
        return asdict(self)

    def resolvable_bands(self) -> dict[str, bool]:
        """Whether each band of :data:`MOTION_BANDS_HZ` lies strictly below the Nyquist limit.

        At 1 Hz sampling even breathing sits on the Nyquist frequency itself,
        where amplitude and phase cannot be told apart.

        Examples
        --------
        >>> d = sampling_diagnostics(np.arange(0, 60, 1.0), max_gap_s=5.0)
        >>> d.resolvable_bands()["breathing"], d.resolvable_bands()["walking Doppler"]
        (False, False)
        """
        return {name: high < self.nyquist_hz for name, (_, high) in MOTION_BANDS_HZ.items()}

    def describe_limits(self) -> list[str]:
        """Plain-language statements of what this sampling supports."""
        statements = [
            f"long-run packet rate {self.mean_rate_hz:.3g} Hz -> Nyquist {self.nyquist_hz:.3g} Hz"
        ]
        for name, (low, high) in MOTION_BANDS_HZ.items():
            if high < self.nyquist_hz:
                verdict = "resolvable"
            elif low < self.nyquist_hz:
                verdict = "only partly resolvable"
            else:
                verdict = "NOT resolvable (aliased)"
            statements.append(f"{name} ({low:g}-{high:g} Hz): {verdict}")
        if self.burst_fraction > 0.2 and self.n_gaps:
            statements.append(
                f"{self.burst_fraction:.0%} of intervals are bursts, with {self.n_gaps} "
                "long pause(s): resample per contiguous segment, not across pauses"
            )
        return statements


def sampling_diagnostics(times_s: np.ndarray, *, max_gap_s: float) -> SamplingDiagnostics:
    """Summarize the sampling of a timeline.

    Parameters
    ----------
    times_s : numpy.ndarray
        Monotone sample times in seconds (at least two samples).
    max_gap_s : float
        Pauses longer than this are counted as gaps.

    Returns
    -------
    SamplingDiagnostics
        The summary.

    Raises
    ------
    ValueError
        If fewer than two samples are given or the times are not sorted.

    Examples
    --------
    >>> d = sampling_diagnostics(np.arange(0, 10, 0.1), max_gap_s=1.0)
    >>> round(d.median_rate_hz, 6), d.n_gaps
    (10.0, 0)
    """
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("sampling diagnostics need at least two timestamps")
    intervals = np.diff(times)
    if np.any(intervals < 0):
        raise ValueError("timestamps must be sorted; unwrap and sort them first")
    duration = float(times[-1] - times[0])
    median_interval = float(np.median(intervals))
    mean_interval = float(np.mean(intervals))
    mean_rate = (times.size - 1) / duration if duration > 0 else float("inf")
    return SamplingDiagnostics(
        n_samples=int(times.size),
        duration_s=duration,
        median_interval_s=median_interval,
        median_rate_hz=1.0 / median_interval if median_interval > 0 else float("inf"),
        mean_rate_hz=mean_rate,
        nyquist_hz=mean_rate / 2.0,
        burst_fraction=float(np.mean(intervals < BURST_INTERVAL_S)),
        interval_p95_s=float(np.percentile(intervals, 95)),
        max_interval_s=float(intervals.max()),
        n_gaps=int(np.count_nonzero(intervals > max_gap_s)),
        interval_cv=float(np.std(intervals) / mean_interval) if mean_interval > 0 else 0.0,
    )
