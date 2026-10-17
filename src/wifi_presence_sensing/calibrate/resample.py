"""Uniform resampling of irregularly timed CSI.

Empirical mode decomposition, the Hilbert transform and every spectral feature
assume samples on a uniform grid.  Packet captures are not: they mix
millisecond bursts with pauses of seconds.  Decomposing such a sequence as if
it were uniform turns the timing pattern itself into "modes".

:func:`resample_uniform` therefore

1. splits the timeline wherever a pause exceeds ``max_gap_s``, since the
   motion may have changed during the pause and nothing is interpolated
   across it;
2. on each contiguous segment, places a uniform grid at ``rate_hz`` and

   * averages all packets falling into each grid cell when the native rate is
     well above the target (``bin_mean``: an anti-aliasing box filter), or
   * interpolates linearly when it is not (``linear``);

3. reports per segment how many grid cells had to be filled by interpolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from ..capture.timeline import contiguous_segments

_LOGGER = logging.getLogger(__name__)

#: Resampling methods accepted by :func:`resample_uniform`.
RESAMPLE_METHODS: tuple[str, ...] = ("auto", "linear", "bin_mean")


@dataclass(frozen=True, eq=False)
class UniformSegment:
    """One uniformly resampled stretch of a trace.

    Attributes
    ----------
    start_s : float
        Time of the first grid point.
    rate_hz : float
        Grid rate.
    values : numpy.ndarray
        Resampled values, time on axis 0.
    source_slice : slice
        Range of input samples the segment was built from.
    method : str
        ``linear`` or ``bin_mean``.
    interpolated_fraction : float
        Fraction of grid points with no input sample in their cell
        (``bin_mean``), filled by interpolation.
    """

    start_s: float
    rate_hz: float
    values: np.ndarray
    source_slice: slice
    method: str
    interpolated_fraction: float

    @property
    def n_samples(self) -> int:
        """Number of grid points."""
        return int(self.values.shape[0])

    @property
    def duration_s(self) -> float:
        """Span of the grid."""
        return (self.n_samples - 1) / self.rate_hz

    @property
    def times_s(self) -> np.ndarray:
        """Grid times."""
        return self.start_s + np.arange(self.n_samples) / self.rate_hz


def _bin_mean(
    times: np.ndarray, values: np.ndarray, n_out: int, rate: float
) -> tuple[np.ndarray, float]:
    """Average the samples falling into each grid cell; interpolate empty cells."""
    cell = np.clip(np.rint((times - times[0]) * rate).astype(np.int64), 0, n_out - 1)
    flat = values.reshape(values.shape[0], -1)
    counts = np.bincount(cell, minlength=n_out).astype(np.float64)
    sums = np.zeros((n_out, flat.shape[1]), dtype=np.result_type(flat.dtype, np.float64))
    np.add.at(sums, cell, flat)
    filled = counts > 0
    out = np.empty_like(sums)
    out[filled] = sums[filled] / counts[filled, None]
    if not filled.all():
        grid = np.arange(n_out, dtype=np.float64)
        known = out[filled]
        if known.shape[0] >= 2:
            fill = interp1d(
                grid[filled],
                known,
                axis=0,
                assume_sorted=True,
                bounds_error=False,
                fill_value=(known[0], known[-1]),
            )
            out[~filled] = fill(grid[~filled])
        else:
            out[~filled] = known[0]
    return out.reshape((n_out, *values.shape[1:])), float(np.mean(~filled))


def resample_uniform(
    times_s: np.ndarray,
    values: np.ndarray,
    rate_hz: float,
    *,
    max_gap_s: float,
    method: str = "auto",
    min_duration_s: float = 0.0,
) -> list[UniformSegment]:
    """Resample an irregular series onto uniform grids, one per contiguous segment.

    Parameters
    ----------
    times_s : numpy.ndarray
        Sorted sample times in seconds.
    values : numpy.ndarray
        Samples with time on axis 0; any trailing shape.
    rate_hz : float
        Target rate.
    max_gap_s : float
        Pauses longer than this split the series.
    method : {"auto", "linear", "bin_mean"}, optional
        ``auto`` uses ``bin_mean`` when the median native interval is shorter
        than half the target interval, ``linear`` otherwise.
    min_duration_s : float, optional
        Segments shorter than this are dropped.

    Returns
    -------
    list of UniformSegment
        Resampled segments in time order.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or the method is unknown.

    Examples
    --------
    >>> t = np.array([0.0, 0.05, 0.31, 0.5, 0.74, 1.0, 5.0, 5.5, 6.0])
    >>> segments = resample_uniform(t, 2.0 * t, 4.0, max_gap_s=1.0, method="linear")
    >>> [s.n_samples for s in segments]
    [5, 5]
    >>> bool(np.allclose(segments[0].values, 2.0 * segments[0].times_s))
    True
    """
    if method not in RESAMPLE_METHODS:
        raise ValueError(f"method={method!r} is not one of {RESAMPLE_METHODS}")
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    times = np.asarray(times_s, dtype=np.float64)
    data = np.asarray(values)
    if times.ndim != 1 or data.shape[0] != times.size:
        raise ValueError("values must have one row per timestamp")
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be sorted")

    segments: list[UniformSegment] = []
    for piece in contiguous_segments(times, max_gap_s, min_samples=2):
        t = times[piece]
        v = data[piece]
        span = t[-1] - t[0]
        if span < max(min_duration_s, 1.0 / rate_hz):
            continue
        n_out = int(np.floor(span * rate_hz + 1e-9)) + 1
        chosen = method
        if chosen == "auto":
            native = float(np.median(np.diff(t)))
            chosen = "bin_mean" if native < 0.5 / rate_hz else "linear"
        if chosen == "bin_mean":
            out, interpolated = _bin_mean(t, v, n_out, rate_hz)
        else:
            # Repeated timestamps make the interpolant ambiguous; keep the first packet.
            unique_t, first = np.unique(t, return_index=True)
            grid = t[0] + np.arange(n_out) / rate_hz
            out = interp1d(unique_t, v[first], axis=0, assume_sorted=True)(
                np.clip(grid, unique_t[0], unique_t[-1])
            )
            interpolated = 0.0
        segments.append(
            UniformSegment(
                start_s=float(t[0]),
                rate_hz=float(rate_hz),
                values=out,
                source_slice=piece,
                method=chosen,
                interpolated_fraction=interpolated,
            )
        )
    if not segments:
        _LOGGER.warning(
            "no segment of at least %.3g s survived gap splitting (max gap %.3g s)",
            max(min_duration_s, 1.0 / rate_hz),
            max_gap_s,
        )
    return segments
