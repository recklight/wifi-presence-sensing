"""Empirical mode decomposition (EMD).

EMD writes a signal as a sum of intrinsic mode functions (IMFs) and a residual
trend (Huang et al., 1998):

.. math::

    x(t) = \\sum_{k=1}^{K} c_k(t) + r(t).

Each IMF is extracted by sifting (:mod:`wifi_presence_sensing.decompose.sifting`) until a
stopping criterion (:mod:`wifi_presence_sensing.decompose.stopping`) accepts it; the IMF is
subtracted and the process repeats on the remainder until the remainder is a
trend (fewer than three extrema) or ``max_modes`` IMFs have been taken.

The decomposition is complete by construction, since the residual is whatever
the IMFs leave over.  :func:`decompose_emd` still checks the sum before it
returns, so a numerical regression raises an error instead of slipping
quietly into the features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .sifting import DEFAULT_MIRRORED_EXTREMA, find_extrema, sift_once
from .stopping import SiftState, StoppingCriterion, ThreeThresholdStopping

_LOGGER = logging.getLogger(__name__)

#: Hard cap on sifting steps per IMF, whatever the criterion says.
DEFAULT_MAX_SIFT_ITERATIONS = 1000


def default_max_modes(n_samples: int) -> int:
    """A practical bound on the number of IMFs, ``floor(log2(n)) - 1``.

    Each IMF roughly halves the dominant frequency, so a record of ``n``
    samples cannot hold many more dyadic modes than this.

    Examples
    --------
    >>> default_max_modes(256)
    7
    """
    return max(1, int(np.floor(np.log2(max(n_samples, 2)))) - 1)


@dataclass(frozen=True, eq=False)
class ModeDecomposition:
    """IMFs and residual of a decomposition.

    Attributes
    ----------
    signal : numpy.ndarray
        The decomposed signal.
    imfs : numpy.ndarray
        Shaped ``(n_modes, n_samples)``, highest frequency first.
    residual : numpy.ndarray
        The trend left over.
    sift_iterations : tuple of int
        Sifting steps spent on each IMF.
    """

    signal: np.ndarray
    imfs: np.ndarray
    residual: np.ndarray
    sift_iterations: tuple[int, ...] = ()

    @property
    def n_modes(self) -> int:
        """Number of IMFs."""
        return int(self.imfs.shape[0])

    def reconstruct(self) -> np.ndarray:
        """Sum of the IMFs and the residual."""
        return self.imfs.sum(axis=0) + self.residual


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {x.shape}")
    if x.size < 4:
        raise ValueError(f"signal needs at least 4 samples, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinite values")
    return x


def extract_mode(
    signal: np.ndarray,
    *,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    time: np.ndarray | None = None,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> tuple[np.ndarray, int] | None:
    """Sift out the highest-frequency IMF of a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional signal.
    stopping : StoppingCriterion, optional
        Defaults to :class:`ThreeThresholdStopping`.
    max_sift_iterations : int, optional
        Hard cap on sifting steps.
    time : numpy.ndarray, optional
        Sample times; defaults to indices.
    n_mirror : int, optional
        Extrema mirrored at each end for the envelopes.

    Returns
    -------
    (numpy.ndarray, int) or None
        The IMF and the number of sifting steps used, or ``None`` when the
        signal is already a trend (fewer than three extrema).
    """
    criterion = ThreeThresholdStopping() if stopping is None else stopping
    if max_sift_iterations < 1:
        raise ValueError(f"max_sift_iterations must be at least 1, got {max_sift_iterations}")
    candidate = np.asarray(signal, dtype=np.float64)
    previous: np.ndarray | None = None
    for iteration in range(max_sift_iterations + 1):
        step = sift_once(candidate, time, n_mirror=n_mirror)
        if step is None:
            if iteration == 0:
                return None
            # Sifting flattened the candidate below three extrema: accept it.
            return candidate, iteration
        state = SiftState(
            candidate=candidate,
            previous=previous,
            mean_envelope=step.envelopes.mean,
            amplitude=step.envelopes.amplitude,
            iteration=iteration,
            n_extrema=step.extrema.count,
            n_zero_crossings=step.n_zero_crossings,
        )
        if iteration == max_sift_iterations or criterion.should_stop(state):
            if iteration == max_sift_iterations:
                _LOGGER.debug("sifting reached the %d-step cap", max_sift_iterations)
            return candidate, iteration
        previous = candidate
        candidate = step.candidate
    return candidate, max_sift_iterations  # pragma: no cover - loop always returns


def decompose_emd(
    signal: np.ndarray,
    *,
    max_modes: int | None = None,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    time: np.ndarray | None = None,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> ModeDecomposition:
    """Decompose a signal into IMFs and a residual trend.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional, finite signal of at least four samples.
    max_modes : int, optional
        Maximum number of IMFs; defaults to :func:`default_max_modes`.
    stopping : StoppingCriterion, optional
        Sifting stopping criterion; defaults to :class:`ThreeThresholdStopping`.
    max_sift_iterations : int, optional
        Hard cap on sifting steps per IMF.
    time : numpy.ndarray, optional
        Sample times (strictly increasing); defaults to indices.
    n_mirror : int, optional
        Extrema mirrored at each end for the envelopes.

    Returns
    -------
    ModeDecomposition
        IMFs (highest frequency first) and residual, which sum exactly to the
        input.

    Raises
    ------
    ValueError
        If the signal is not one-dimensional, too short, or not finite.

    Examples
    --------
    >>> t = np.linspace(0, 1, 512)
    >>> x = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 4 * t) + t
    >>> result = decompose_emd(x)
    >>> bool(np.allclose(result.reconstruct(), x))
    True
    >>> bool(np.corrcoef(result.imfs[0], np.sin(2 * np.pi * 40 * t))[0, 1] > 0.99)
    True
    """
    x = _validate_signal(signal)
    limit = default_max_modes(x.size) if max_modes is None else int(max_modes)
    if limit < 0:
        raise ValueError(f"max_modes must be non-negative, got {max_modes}")
    if time is not None and (np.shape(time) != x.shape or np.any(np.diff(time) <= 0)):
        raise ValueError("time must be strictly increasing with one entry per sample")

    scale = float(np.std(x))
    remainder = x.copy()
    imfs: list[np.ndarray] = []
    iterations: list[int] = []
    while len(imfs) < limit:
        if scale == 0.0 or float(np.std(remainder)) <= 1e-12 * scale:
            break
        if find_extrema(remainder).count < 3:
            break
        extracted = extract_mode(
            remainder,
            stopping=stopping,
            max_sift_iterations=max_sift_iterations,
            time=time,
            n_mirror=n_mirror,
        )
        if extracted is None:
            break
        mode, n_steps = extracted
        imfs.append(mode)
        iterations.append(n_steps)
        remainder = remainder - mode

    stacked = np.vstack(imfs) if imfs else np.zeros((0, x.size))
    residual = x - stacked.sum(axis=0)
    result = ModeDecomposition(
        signal=x, imfs=stacked, residual=residual, sift_iterations=tuple(iterations)
    )
    if not np.allclose(result.reconstruct(), x, rtol=1e-9, atol=1e-9 * max(scale, 1.0)):
        raise RuntimeError("EMD reconstruction check failed: IMFs + residual != signal")
    return result
