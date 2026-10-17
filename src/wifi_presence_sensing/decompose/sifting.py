"""The sifting step of empirical mode decomposition.

One sifting step estimates the local mean of a signal and subtracts it.  The
local mean is the average of two cubic-spline envelopes, one through the local
maxima and one through the local minima (Huang et al., 1998):

.. math::

    m(t) = \\tfrac12\\big(e_{\\max}(t) + e_{\\min}(t)\\big), \\qquad h(t) = x(t) - m(t).

The splines need support beyond the first and last extremum or they swing
wildly at the ends of the record.  Following Rilling et al. (2003), a few
extrema near each end are mirrored about the end extremum (about the end
sample itself when the signal ends beyond its last extremum), so the envelopes
are interpolated, not extrapolated, over the whole record.

References
----------
N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum
for nonlinear and non-stationary time series analysis," Proceedings of the
Royal Society of London A, 454(1971):903-995, 1998.

G. Rilling, P. Flandrin, P. Goncalves, "On Empirical Mode Decomposition and its
Algorithms," IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing,
2003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

#: Number of extrema mirrored beyond each end of the record.
DEFAULT_MIRRORED_EXTREMA = 2


@dataclass(frozen=True, eq=False)
class Extrema:
    """Indices of the local maxima and minima of a signal.

    Plateaus count as a single extremum located at their centre.
    """

    maxima: np.ndarray
    minima: np.ndarray

    @property
    def count(self) -> int:
        """Total number of extrema."""
        return int(self.maxima.size + self.minima.size)


def find_extrema(signal: np.ndarray) -> Extrema:
    """Locate local maxima and minima, treating plateaus as one extremum.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional real signal.

    Returns
    -------
    Extrema
        Sorted index arrays.  End samples are never extrema.

    Examples
    --------
    >>> e = find_extrema(np.array([0.0, 2.0, 1.0, 1.0, 1.0, 3.0, 0.0]))
    >>> e.maxima.tolist(), e.minima.tolist()
    ([1, 5], [3])
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {x.shape}")
    empty = np.zeros(0, dtype=np.int64)
    if x.size < 3:
        return Extrema(empty, empty)
    # Collapse plateaus: keep the first sample of every run of equal values.
    run_start = np.flatnonzero(np.r_[True, np.diff(x) != 0])
    run_end = np.r_[run_start[1:] - 1, x.size - 1]
    centre = (run_start + run_end) // 2
    values = x[run_start]
    if values.size < 3:
        return Extrema(empty, empty)
    step = np.diff(values)
    rising = step[:-1] > 0
    falling = step[1:] < 0
    maxima = centre[1:-1][rising & falling]
    minima = centre[1:-1][~rising & ~falling]
    return Extrema(maxima.astype(np.int64), minima.astype(np.int64))


def count_zero_crossings(signal: np.ndarray) -> int:
    """Number of sign changes, ignoring exact zeros.

    Examples
    --------
    >>> count_zero_crossings(np.array([1.0, -1.0, 0.0, -2.0, 3.0]))
    2
    """
    sign = np.sign(np.asarray(signal, dtype=np.float64))
    sign = sign[sign != 0]
    return int(np.count_nonzero(sign[1:] != sign[:-1]))


def _mirror_start(
    t: np.ndarray, x: np.ndarray, maxima: np.ndarray, minima: np.ndarray, n_mirror: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirrored extrema before the start of the record: ``(t_max, x_max, t_min, x_min)``."""
    if maxima[0] < minima[0]:
        if x[0] > x[minima[0]]:
            sel_max, sel_min, axis = maxima[1 : n_mirror + 1], minima[:n_mirror], maxima[0]
        else:
            sel_max, sel_min, axis = maxima[:n_mirror], np.r_[0, minima[: n_mirror - 1]], 0
    else:
        if x[0] < x[maxima[0]]:
            sel_max, sel_min, axis = maxima[:n_mirror], minima[1 : n_mirror + 1], minima[0]
        else:
            sel_max, sel_min, axis = np.r_[0, maxima[: n_mirror - 1]], minima[:n_mirror], 0

    t_max = 2.0 * t[axis] - t[sel_max]
    t_min = 2.0 * t[axis] - t[sel_min]
    covered = t_max.size > 0 and t_min.size > 0 and t_max.min() <= t[0] and t_min.min() <= t[0]
    if not covered and axis != 0:
        # Mirroring about the end extremum does not reach past the record start:
        # mirror about the first sample instead.
        if axis == maxima[0]:
            sel_max = maxima[:n_mirror]
        else:
            sel_min = minima[:n_mirror]
        t_max = 2.0 * t[0] - t[sel_max]
        t_min = 2.0 * t[0] - t[sel_min]
    return t_max[::-1], x[sel_max][::-1], t_min[::-1], x[sel_min][::-1]


def _envelope_knots(
    t: np.ndarray, x: np.ndarray, extrema: Extrema, n_mirror: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Knot times and values of the upper and lower envelopes, mirrors included."""
    n = x.size
    left = _mirror_start(t, x, extrema.maxima, extrema.minima, n_mirror)
    # The end of the record is the start of the time-reversed record.
    t_rev = -t[::-1]
    right = _mirror_start(
        t_rev,
        x[::-1],
        (n - 1 - extrema.maxima)[::-1],
        (n - 1 - extrema.minima)[::-1],
        n_mirror,
    )
    t_max = np.concatenate([left[0], t[extrema.maxima], -right[0][::-1]])
    x_max = np.concatenate([left[1], x[extrema.maxima], right[1][::-1]])
    t_min = np.concatenate([left[2], t[extrema.minima], -right[2][::-1]])
    x_min = np.concatenate([left[3], x[extrema.minima], right[3][::-1]])
    return (*_strictly_increasing(t_max, x_max), *_strictly_increasing(t_min, x_min))


def _strictly_increasing(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times.size > 1 and np.all(np.diff(times) > 0):
        return times, values
    unique_t, first = np.unique(times, return_index=True)
    return unique_t, values[first]


def natural_cubic_spline(knots: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic spline through ``(knots, values)`` at ``query``.

    The sifting loop builds two splines per step, thousands of times per
    ensemble decomposition, so the spline is computed directly with as little
    overhead as possible: one tridiagonal solve for the knot second derivatives
    :math:`M_i` (zero at both ends), then a vectorized evaluation of

    .. math::

        S(t) = M_i \\frac{(t_{i+1}-t)^3}{6h_i} + M_{i+1}\\frac{(t-t_i)^3}{6h_i}
             + \\Big(y_i - \\frac{M_i h_i^2}{6}\\Big)\\frac{t_{i+1}-t}{h_i}
             + \\Big(y_{i+1} - \\frac{M_{i+1} h_i^2}{6}\\Big)\\frac{t-t_i}{h_i}.

    The end conditions only act beyond the mirrored extrema, i.e. outside the
    record, so the natural condition is as good as any there.

    Parameters
    ----------
    knots : numpy.ndarray
        Strictly increasing knot positions (at least two).
    values : numpy.ndarray
        Values at the knots.
    query : numpy.ndarray
        Evaluation points.

    Returns
    -------
    numpy.ndarray
        Spline values at ``query``.

    Examples
    --------
    >>> k = np.linspace(0, 2 * np.pi, 40)
    >>> q = np.linspace(0.5, 5.5, 7)
    >>> bool(np.allclose(natural_cubic_spline(k, np.sin(k), q), np.sin(q), atol=1e-4))
    True
    """
    t = np.asarray(knots, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    m = t.size
    h = np.diff(t)
    second = np.zeros(m)
    if m > 2:
        slope = np.diff(y) / h
        rhs = 6.0 * np.diff(slope)
        bands = np.zeros((3, m - 2))
        bands[0, 1:] = h[1:-1]
        bands[1] = 2.0 * (h[:-1] + h[1:])
        bands[2, :-1] = h[1:-1]
        second[1:-1] = solve_banded((1, 1), bands, rhs, check_finite=False)
    points = np.asarray(query, dtype=np.float64)
    i = np.clip(np.searchsorted(t, points, side="right") - 1, 0, m - 2)
    width = h[i]
    left = t[i + 1] - points
    right = points - t[i]
    return (
        second[i] * left**3 / (6.0 * width)
        + second[i + 1] * right**3 / (6.0 * width)
        + (y[i] - second[i] * width**2 / 6.0) * left / width
        + (y[i + 1] - second[i + 1] * width**2 / 6.0) * right / width
    )


@dataclass(frozen=True, eq=False)
class EnvelopePair:
    """Upper and lower envelopes of a signal."""

    upper: np.ndarray
    lower: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        """Local mean, the average of the two envelopes."""
        return 0.5 * (self.upper + self.lower)

    @property
    def amplitude(self) -> np.ndarray:
        """Local amplitude, half the distance between the envelopes."""
        return 0.5 * (self.upper - self.lower)


def extrema_envelopes(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
    extrema: Extrema | None = None,
) -> EnvelopePair | None:
    """Cubic-spline envelopes through the maxima and the minima.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional real signal.
    time : numpy.ndarray, optional
        Sample times (strictly increasing); defaults to sample indices.
    n_mirror : int, optional
        Extrema mirrored beyond each end.
    extrema : Extrema, optional
        Precomputed extrema of ``signal``.

    Returns
    -------
    EnvelopePair or None
        ``None`` when the signal has fewer than three extrema in total: it is
        then a trend, not an oscillation, and has no envelopes.
    """
    x = np.asarray(signal, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64) if time is None else np.asarray(time, np.float64)
    found = find_extrema(x) if extrema is None else extrema
    if found.count < 3 or found.maxima.size == 0 or found.minima.size == 0:
        return None
    t_max, x_max, t_min, x_min = _envelope_knots(t, x, found, max(1, int(n_mirror)))
    if t_max.size < 2 or t_min.size < 2:
        return None
    upper = natural_cubic_spline(t_max, x_max, t)
    lower = natural_cubic_spline(t_min, x_min, t)
    return EnvelopePair(upper=upper, lower=lower)


@dataclass(frozen=True, eq=False)
class SiftStep:
    """Outcome of one sifting step.

    Attributes
    ----------
    candidate : numpy.ndarray
        ``signal - mean``: the refined IMF candidate.
    envelopes : EnvelopePair
        The envelopes of the input.
    extrema : Extrema
        The extrema of the input.
    n_zero_crossings : int
        Zero crossings of the input.
    """

    candidate: np.ndarray
    envelopes: EnvelopePair
    extrema: Extrema
    n_zero_crossings: int


def sift_once(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> SiftStep | None:
    """Subtract the envelope mean once.

    Returns
    -------
    SiftStep or None
        ``None`` when the signal has too few extrema to be sifted.

    Examples
    --------
    Sifting removes a slow offset riding under a fast oscillation:

    >>> t = np.linspace(0, 1, 400)
    >>> x = np.sin(2 * np.pi * 20 * t) + 0.5
    >>> step = sift_once(x)
    >>> bool(abs(step.candidate[50:-50].mean()) < 0.05)
    True
    """
    x = np.asarray(signal, dtype=np.float64)
    extrema = find_extrema(x)
    envelopes = extrema_envelopes(x, time, n_mirror=n_mirror, extrema=extrema)
    if envelopes is None:
        return None
    return SiftStep(
        candidate=x - envelopes.mean,
        envelopes=envelopes,
        extrema=extrema,
        n_zero_crossings=count_zero_crossings(x),
    )
