"""Complete ensemble EMD with adaptive noise (CEEMDAN).

EEMD leaves two loose ends: the average of noisy decompositions is not an
exact decomposition of the clean signal, and different members can produce
different numbers of modes, so "mode k" is averaged over unlike things.
CEEMDAN (Torres et al., 2011) extracts the modes *one at a time*, each as an
ensemble average, from a single shared residue:

.. math::

    \\tilde c_1 = \\frac1N\\sum_i E_1\\big(x + \\beta_0 w_i\\big), \\qquad r_1 = x - \\tilde c_1,

    \\tilde c_{k+1} = \\frac1N\\sum_i E_1\\big(r_k + \\beta_k E_k(w_i)\\big), \\qquad
    r_{k+1} = r_k - \\tilde c_{k+1},

where :math:`E_k(\\cdot)` is the k-th mode obtained by EMD and :math:`w_i` is
unit white noise.  Every stage shares one residue, so the decomposition is
complete and all modes are defined for every member.

Noise amplitudes.  Stage 0 adds noise of standard deviation
``noise_scale * std(x)``.  At stage k the noise mode :math:`E_k(w_i)` is first
normalized to unit standard deviation and then scaled to
``noise_scale * std(r_k)``, so the added noise always keeps the same ratio to
the residue it perturbs.  This follows the residue-proportional amplitudes
:math:`\\beta_k = \\varepsilon_0\\,\\sigma(r_k)` of Colominas et al. (2014); unlike
there, the noise mode itself is normalized, so the ratio does not shrink with
the (decreasing) power of the higher-order noise modes.

References
----------
M. E. Torres, M. A. Colominas, G. Schlotthauer, P. Flandrin, "A Complete
Ensemble Empirical Mode Decomposition with Adaptive Noise," IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), 2011.

M. A. Colominas, G. Schlotthauer, M. E. Torres, "Improved complete ensemble
EMD: A suitable tool for biomedical signal processing," Biomedical Signal
Processing and Control, 14:19-29, 2014.
"""

from __future__ import annotations

import logging

import numpy as np
from joblib import Parallel, delayed

from .eemd import EnsembleDecomposition, member_noise
from .emd import (
    DEFAULT_MAX_SIFT_ITERATIONS,
    _validate_signal,
    decompose_emd,
    default_max_modes,
    extract_mode,
)
from .sifting import DEFAULT_MIRRORED_EXTREMA, find_extrema
from .stopping import StoppingCriterion, ThreeThresholdStopping

_LOGGER = logging.getLogger(__name__)


def _first_mode(
    signal: np.ndarray, stopping: StoppingCriterion, max_sift_iterations: int, n_mirror: int
) -> np.ndarray:
    """First IMF of a signal, or zeros when the signal is already a trend."""
    extracted = extract_mode(
        signal, stopping=stopping, max_sift_iterations=max_sift_iterations, n_mirror=n_mirror
    )
    return np.zeros_like(signal) if extracted is None else extracted[0]


def _noise_modes(
    noise: np.ndarray,
    n_modes: int,
    stopping: StoppingCriterion,
    max_sift_iterations: int,
    n_mirror: int,
) -> np.ndarray:
    """EMD modes of one noise realization, padded with zeros to ``n_modes``."""
    result = decompose_emd(
        noise,
        max_modes=n_modes,
        stopping=stopping,
        max_sift_iterations=max_sift_iterations,
        n_mirror=n_mirror,
    )
    padded = np.zeros((n_modes, noise.size))
    padded[: result.n_modes] = result.imfs
    return padded


def decompose_ceemdan(
    signal: np.ndarray,
    *,
    n_ensemble: int = 100,
    noise_scale: float = 0.2,
    max_modes: int | None = None,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    seed: int | None = None,
    n_jobs: int = 1,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> EnsembleDecomposition:
    """Complete ensemble EMD with adaptive noise.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional, finite signal.
    n_ensemble : int, optional
        Noise realizations per stage.
    noise_scale : float, optional
        Relative noise level of every stage.
    max_modes : int, optional
        Maximum number of IMFs; defaults to :func:`default_max_modes`.
    stopping : StoppingCriterion, optional
        Defaults to :class:`ThreeThresholdStopping`.
    max_sift_iterations : int, optional
        Hard cap on sifting steps per IMF.
    seed : int, optional
        Seed of the noise generator.
    n_jobs : int, optional
        Parallel workers per stage.  Does not change the result.
    n_mirror : int, optional
        Extrema mirrored at each end for the envelopes.

    Returns
    -------
    EnsembleDecomposition
        IMFs and residual summing exactly to the signal (``method="ceemdan"``).

    Examples
    --------
    >>> t = np.linspace(0, 2, 256)
    >>> x = np.sin(2 * np.pi * 12 * t) + np.sin(2 * np.pi * 1.5 * t)
    >>> result = decompose_ceemdan(x, n_ensemble=8, seed=1)
    >>> bool(np.allclose(result.reconstruct(), x))
    True
    """
    x = _validate_signal(signal)
    if n_ensemble < 1:
        raise ValueError(f"n_ensemble must be at least 1, got {n_ensemble}")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    if seed is None:
        _LOGGER.warning("decompose_ceemdan called without a seed: the result is not reproducible")
    n_modes = default_max_modes(x.size) if max_modes is None else int(max_modes)
    criterion = ThreeThresholdStopping() if stopping is None else stopping

    def run(tasks: list[tuple]) -> list[np.ndarray]:
        if n_jobs == 1:
            return [function(*args) for function, *args in tasks]
        return Parallel(n_jobs=n_jobs)(delayed(function)(*args) for function, *args in tasks)

    noise = member_noise(seed, n_ensemble, x.size)
    noise_modes = np.stack(
        run(
            [
                (_noise_modes, row, n_modes, criterion, max_sift_iterations, n_mirror)
                for row in noise
            ]
        )
    )  # (members, modes, samples)

    sigma_x = float(np.std(x))
    residue = x.copy()
    imfs: list[np.ndarray] = []
    for stage in range(n_modes):
        if find_extrema(residue).count < 3 or float(np.std(residue)) <= 1e-12 * max(
            sigma_x, 1e-300
        ):
            break
        if stage == 0:
            perturbations = noise * noise_scale * sigma_x
        else:
            component = noise_modes[:, stage - 1, :]
            spread = component.std(axis=1, keepdims=True)
            spread[spread == 0] = 1.0
            perturbations = component / spread * noise_scale * float(np.std(residue))
        members = run(
            [
                (_first_mode, residue + perturbation, criterion, max_sift_iterations, n_mirror)
                for perturbation in perturbations
            ]
        )
        mode = np.mean(members, axis=0)
        if not np.any(mode):
            break
        imfs.append(mode)
        residue = residue - mode

    stacked = np.vstack(imfs) if imfs else np.zeros((0, x.size))
    residual = x - stacked.sum(axis=0)
    return EnsembleDecomposition(
        signal=x,
        imfs=stacked,
        residual=residual,
        n_members=n_ensemble,
        noise_scale=float(noise_scale),
        mode_counts=np.full(stacked.shape[0], n_ensemble),
        residual_noise=np.zeros(x.size),
        seed=seed,
        method="ceemdan",
    )
