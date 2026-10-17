"""Empirical mode decomposition: EMD, EEMD and CEEMDAN with pluggable stopping."""

from __future__ import annotations

import numpy as np

from .ceemdan import decompose_ceemdan
from .eemd import EnsembleDecomposition, decompose_eemd, member_noise
from .emd import (
    DEFAULT_MAX_SIFT_ITERATIONS,
    ModeDecomposition,
    decompose_emd,
    default_max_modes,
    extract_mode,
)
from .sifting import (
    EnvelopePair,
    Extrema,
    SiftStep,
    count_zero_crossings,
    extrema_envelopes,
    find_extrema,
    sift_once,
)
from .stopping import (
    STOPPING_CRITERIA,
    CauchyStopping,
    FixedIterationStopping,
    SiftState,
    StoppingCriterion,
    ThreeThresholdStopping,
    stopping_from_name,
)

#: Decomposition methods accepted by :func:`decompose`.
DECOMPOSITION_METHODS: tuple[str, ...] = ("emd", "eemd", "ceemdan")


def decompose(
    signal: np.ndarray,
    method: str = "eemd",
    *,
    n_ensemble: int = 100,
    noise_scale: float = 0.2,
    max_modes: int | None = None,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    seed: int | None = None,
    n_jobs: int = 1,
    paired_noise: bool = False,
) -> ModeDecomposition | EnsembleDecomposition:
    """Dispatch to :func:`decompose_emd`, :func:`decompose_eemd` or :func:`decompose_ceemdan`.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional signal.
    method : {"emd", "eemd", "ceemdan"}, optional
        Decomposition method.  Ensemble parameters are ignored by ``emd``.
    n_ensemble, noise_scale, max_modes, stopping, max_sift_iterations, seed, n_jobs
        Forwarded to the chosen method.
    paired_noise : bool, optional
        EEMD only: use each noise realization with both signs.

    Returns
    -------
    ModeDecomposition or EnsembleDecomposition
        Both expose ``imfs``, ``residual`` and ``reconstruct()``.

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 20, 128)) + np.linspace(0, 1, 128)
    >>> decompose(x, "emd").imfs.shape[1]
    128
    """
    if method == "emd":
        return decompose_emd(
            signal, max_modes=max_modes, stopping=stopping, max_sift_iterations=max_sift_iterations
        )
    if method == "eemd":
        return decompose_eemd(
            signal,
            n_ensemble=n_ensemble,
            noise_scale=noise_scale,
            max_modes=max_modes,
            stopping=stopping,
            max_sift_iterations=max_sift_iterations,
            seed=seed,
            n_jobs=n_jobs,
            paired_noise=paired_noise,
        )
    if method == "ceemdan":
        return decompose_ceemdan(
            signal,
            n_ensemble=n_ensemble,
            noise_scale=noise_scale,
            max_modes=max_modes,
            stopping=stopping,
            max_sift_iterations=max_sift_iterations,
            seed=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(
        f"unknown decomposition method {method!r}; choose one of {DECOMPOSITION_METHODS}"
    )


__all__ = [
    "DECOMPOSITION_METHODS",
    "DEFAULT_MAX_SIFT_ITERATIONS",
    "STOPPING_CRITERIA",
    "CauchyStopping",
    "EnsembleDecomposition",
    "EnvelopePair",
    "Extrema",
    "FixedIterationStopping",
    "ModeDecomposition",
    "SiftState",
    "SiftStep",
    "StoppingCriterion",
    "ThreeThresholdStopping",
    "count_zero_crossings",
    "decompose",
    "decompose_ceemdan",
    "decompose_eemd",
    "decompose_emd",
    "default_max_modes",
    "extract_mode",
    "extrema_envelopes",
    "find_extrema",
    "member_noise",
    "sift_once",
    "stopping_from_name",
]
