"""Ensemble empirical mode decomposition (EEMD).

Plain EMD suffers from mode mixing: an intermittent component jumps between
IMFs, so "IMF 2" means different time scales in different parts of the
record.  EEMD (Wu & Huang, 2009) adds white noise to give every scale a
uniform reference, decomposes many noisy copies, and averages mode by mode:

.. math::

    x_i(t) = x(t) + \\varepsilon\\,\\sigma_x\\,w_i(t), \\qquad
    \\bar c_k(t) = \\frac{1}{N}\\sum_{i=1}^{N} c_{k}^{(i)}(t),

with :math:`w_i` independent unit white noise.  The added noise averages out at
a rate of :math:`\\varepsilon / \\sqrt{N}`.  With ``paired_noise`` each noise
realization is also used with the opposite sign (Yeh et al., 2010), so the
noise cancels exactly in the sum of all modes.

Every member decomposes into the same number of modes (missing ones are zero)
so the average is taken over like with like.  The residual is defined as
``x - sum(mean IMFs)``; the reconstruction of the *clean* signal is therefore
exact, and the part of the ensemble noise that did not average out is reported
separately as :attr:`EnsembleDecomposition.residual_noise`.

Reproducibility.  Each member draws its noise from its own child of
``numpy.random.SeedSequence(seed)``, so the result depends only on ``seed``,
not on ``n_jobs`` or on the order in which workers finish.

References
----------
Z. Wu, N. E. Huang, "Ensemble Empirical Mode Decomposition: A Noise-Assisted
Data Analysis Method," Advances in Adaptive Data Analysis, 1(1):1-41, 2009.

J.-R. Yeh, J.-S. Shieh, N. E. Huang, "Complementary Ensemble Empirical Mode
Decomposition: A Novel Noise Enhanced Data Analysis Method," Advances in
Adaptive Data Analysis, 2(2):135-156, 2010.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from .emd import (
    DEFAULT_MAX_SIFT_ITERATIONS,
    _validate_signal,
    decompose_emd,
    default_max_modes,
)
from .sifting import DEFAULT_MIRRORED_EXTREMA
from .stopping import StoppingCriterion, ThreeThresholdStopping

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class EnsembleDecomposition:
    """Result of a noise-assisted decomposition.

    Attributes
    ----------
    signal : numpy.ndarray
        The decomposed signal.
    imfs : numpy.ndarray
        Ensemble-mean IMFs, ``(n_modes, n_samples)``, highest frequency first.
    residual : numpy.ndarray
        ``signal - imfs.sum(axis=0)``.
    n_members : int
        Number of decompositions averaged.
    noise_scale : float
        Added noise standard deviation relative to the signal's.
    mode_counts : numpy.ndarray
        For each mode, how many members actually produced it (the rest
        contributed zeros).
    residual_noise : numpy.ndarray
        Mean of the noise added across members: what did not average out.
        CEEMDAN does not track it and leaves it at zero.
    seed : int or None
        Seed of the noise generator.
    method : str
        ``"eemd"`` or ``"ceemdan"``.
    """

    signal: np.ndarray
    imfs: np.ndarray
    residual: np.ndarray
    n_members: int
    noise_scale: float
    mode_counts: np.ndarray
    residual_noise: np.ndarray
    seed: int | None
    method: str = "eemd"

    @property
    def n_modes(self) -> int:
        """Number of IMFs."""
        return int(self.imfs.shape[0])

    def reconstruct(self) -> np.ndarray:
        """Sum of the IMFs and the residual (equals ``signal``)."""
        return self.imfs.sum(axis=0) + self.residual


def _member(
    signal: np.ndarray,
    noise: np.ndarray,
    n_modes: int,
    stopping: StoppingCriterion,
    max_sift_iterations: int,
    n_mirror: int,
) -> tuple[np.ndarray, int]:
    """Decompose one noisy copy; return its IMFs padded to ``n_modes``."""
    result = decompose_emd(
        signal + noise,
        max_modes=n_modes,
        stopping=stopping,
        max_sift_iterations=max_sift_iterations,
        n_mirror=n_mirror,
    )
    padded = np.zeros((n_modes, signal.size))
    padded[: result.n_modes] = result.imfs
    return padded, result.n_modes


def member_noise(seed: int | None, n_realizations: int, n_samples: int) -> np.ndarray:
    """Unit-variance white noise, one row per realization, from independent child streams.

    Examples
    --------
    >>> a = member_noise(7, 3, 5)
    >>> b = member_noise(7, 3, 5)
    >>> a.shape, bool(np.array_equal(a, b))
    ((3, 5), True)
    """
    children = np.random.SeedSequence(seed).spawn(n_realizations)
    return np.stack([np.random.default_rng(child).standard_normal(n_samples) for child in children])


def decompose_eemd(
    signal: np.ndarray,
    *,
    n_ensemble: int = 100,
    noise_scale: float = 0.2,
    max_modes: int | None = None,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    seed: int | None = None,
    n_jobs: int = 1,
    paired_noise: bool = False,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> EnsembleDecomposition:
    """Ensemble empirical mode decomposition.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional, finite signal.
    n_ensemble : int, optional
        Number of noise realizations (100 is the customary choice).
    noise_scale : float, optional
        Noise standard deviation relative to the signal's (0.1-0.2 typical).
    max_modes : int, optional
        IMFs per member; defaults to :func:`default_max_modes`.
    stopping : StoppingCriterion, optional
        Defaults to :class:`~wifi_presence_sensing.decompose.stopping.ThreeThresholdStopping`;
        ``FixedIterationStopping(10)`` reproduces the fixed sifting budget
        used by Wu & Huang.
    max_sift_iterations : int, optional
        Hard cap on sifting steps per IMF.
    seed : int, optional
        Seed of the noise generator.  ``None`` draws fresh entropy (the run is
        then not reproducible; a warning is logged).
    n_jobs : int, optional
        Parallel workers (``-1`` for all cores).  Does not change the result.
    paired_noise : bool, optional
        Also decompose each realization with the opposite sign.
    n_mirror : int, optional
        Extrema mirrored at each end for the envelopes.

    Returns
    -------
    EnsembleDecomposition
        Ensemble-mean IMFs and residual.

    Raises
    ------
    ValueError
        If a parameter is out of range or the signal is unusable.

    Examples
    --------
    >>> t = np.linspace(0, 2, 256)
    >>> x = np.sin(2 * np.pi * 12 * t) + np.sin(2 * np.pi * 1.5 * t)
    >>> result = decompose_eemd(x, n_ensemble=10, seed=3)
    >>> bool(np.allclose(result.reconstruct(), x))
    True
    >>> again = decompose_eemd(x, n_ensemble=10, seed=3)
    >>> bool(np.array_equal(result.imfs, again.imfs))
    True
    """
    x = _validate_signal(signal)
    if n_ensemble < 1:
        raise ValueError(f"n_ensemble must be at least 1, got {n_ensemble}")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    if seed is None:
        _LOGGER.warning("decompose_eemd called without a seed: the result is not reproducible")
    n_modes = default_max_modes(x.size) if max_modes is None else int(max_modes)
    criterion = ThreeThresholdStopping() if stopping is None else stopping

    sigma = float(np.std(x))
    noise = member_noise(seed, n_ensemble, x.size) * noise_scale * sigma
    if paired_noise:
        noise = np.concatenate([noise, -noise])

    arguments = [(x, row, n_modes, criterion, max_sift_iterations, n_mirror) for row in noise]
    if n_jobs == 1:
        outputs = [_member(*args) for args in arguments]
    else:
        outputs = Parallel(n_jobs=n_jobs)(delayed(_member)(*args) for args in arguments)

    stacked = np.stack([imfs for imfs, _ in outputs])  # (members, modes, samples)
    produced = np.array([count for _, count in outputs])
    mode_counts = (produced[:, None] > np.arange(n_modes)[None, :]).sum(axis=0)
    mean_imfs = stacked.mean(axis=0)
    # Drop trailing modes that no member produced.
    keep = int(np.max(np.flatnonzero(mode_counts > 0)) + 1) if mode_counts.any() else 0
    mean_imfs = mean_imfs[:keep]
    residual = x - mean_imfs.sum(axis=0)
    return EnsembleDecomposition(
        signal=x,
        imfs=mean_imfs,
        residual=residual,
        n_members=int(noise.shape[0]),
        noise_scale=float(noise_scale),
        mode_counts=mode_counts[:keep],
        residual_noise=noise.mean(axis=0),
        seed=seed,
        method="eemd",
    )
