"""Pluggable stopping criteria for the sifting loop.

Sifting refines an IMF candidate until it is "good enough", and where that line
sits changes the result: too little sifting leaves riding waves, too much
flattens the amplitude modulation that carries the information.  Each
criterion here is a small immutable object with one method,
``should_stop(state)``, so a decomposition can be configured and reported by
name.

``ThreeThresholdStopping`` (default)
    Rilling et al. (2003).  With the evaluation function
    :math:`\\sigma(t) = |m(t)| / a(t)` (envelope mean over envelope
    amplitude), stop when :math:`\\sigma < \\theta_1` on all but a fraction
    :math:`\\alpha` of the record and :math:`\\sigma < \\theta_2` everywhere.
``CauchyStopping``
    Huang et al. (1998), the normalized squared difference between successive
    candidates :math:`\\sum (h_{k-1}-h_k)^2 / \\sum h_{k-1}^2 < \\varepsilon`.
``FixedIterationStopping``
    A fixed number of sifting steps, the choice of Wu & Huang (2009) for the
    ensemble method, where it keeps every member's IMFs on a comparable
    dyadic filter bank.

The first two also require the IMF condition: the numbers of extrema and of
zero crossings differ by at most one.  Independently of the criterion, the
decomposition caps the number of sifting steps (``max_sift_iterations``) so a
pathological signal can never loop forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

#: Names accepted by :func:`stopping_from_name`.
STOPPING_CRITERIA: tuple[str, ...] = ("three_threshold", "cauchy", "fixed")


@dataclass(frozen=True, eq=False)
class SiftState:
    """Everything a stopping criterion may look at.

    Attributes
    ----------
    candidate : numpy.ndarray
        Current IMF candidate :math:`h_k`.
    previous : numpy.ndarray or None
        Previous candidate :math:`h_{k-1}`, ``None`` on the first step.
    mean_envelope : numpy.ndarray
        Envelope mean of ``candidate``.
    amplitude : numpy.ndarray
        Half the envelope gap of ``candidate``.
    iteration : int
        Sifting steps already applied to obtain ``candidate``.
    n_extrema : int
        Extrema of ``candidate``.
    n_zero_crossings : int
        Zero crossings of ``candidate``.
    """

    candidate: np.ndarray
    previous: np.ndarray | None
    mean_envelope: np.ndarray
    amplitude: np.ndarray
    iteration: int
    n_extrema: int
    n_zero_crossings: int

    @property
    def satisfies_imf_condition(self) -> bool:
        """Extrema and zero crossings differ by at most one."""
        return abs(self.n_extrema - self.n_zero_crossings) <= 1


@runtime_checkable
class StoppingCriterion(Protocol):
    """Interface of a sifting stopping criterion."""

    def should_stop(self, state: SiftState) -> bool:
        """Return True when ``state.candidate`` is accepted as an IMF."""
        ...


@dataclass(frozen=True)
class ThreeThresholdStopping:
    """Three-threshold criterion of Rilling, Flandrin and Goncalves (2003).

    Parameters
    ----------
    theta1 : float
        Threshold that must hold on all but a fraction ``alpha`` of samples.
    theta2 : float
        Threshold that must hold everywhere (conventionally ``10 * theta1``).
    alpha : float
        Tolerated fraction of samples above ``theta1``.

    Examples
    --------
    >>> t = np.linspace(0, 1, 200)
    >>> h = np.sin(2 * np.pi * 8 * t)
    >>> state = SiftState(h, None, np.zeros_like(h), np.ones_like(h), 0, 16, 15)
    >>> ThreeThresholdStopping().should_stop(state)
    True
    """

    theta1: float = 0.05
    theta2: float = 0.5
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.theta1 < self.theta2:
            raise ValueError(f"need 0 < theta1 < theta2, got {self.theta1}, {self.theta2}")
        if not 0 <= self.alpha < 1:
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")

    def should_stop(self, state: SiftState) -> bool:
        """Accept the candidate when the envelope mean is small relative to the amplitude."""
        if not state.satisfies_imf_condition:
            return False
        amplitude = np.abs(state.amplitude)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(amplitude > 0, np.abs(state.mean_envelope) / amplitude, np.inf)
        return bool(np.mean(sigma > self.theta1) < self.alpha and np.all(sigma < self.theta2))


@dataclass(frozen=True)
class CauchyStopping:
    """Cauchy-type convergence criterion on successive candidates (Huang et al., 1998).

    Parameters
    ----------
    tolerance : float
        Threshold on :math:`\\sum (h_{k-1}-h_k)^2 / \\sum h_{k-1}^2`.
    require_imf_condition : bool
        Also require the extrema / zero-crossing condition.
    """

    tolerance: float = 0.2
    require_imf_condition: bool = True

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")

    def should_stop(self, state: SiftState) -> bool:
        """Accept the candidate once successive candidates stop changing."""
        if state.previous is None:
            return False
        if self.require_imf_condition and not state.satisfies_imf_condition:
            return False
        energy = float(np.sum(state.previous**2))
        if energy == 0.0:
            return True
        change = float(np.sum((state.previous - state.candidate) ** 2))
        return change / energy < self.tolerance


@dataclass(frozen=True)
class FixedIterationStopping:
    """Stop after a fixed number of sifting steps (Wu & Huang, 2009).

    Parameters
    ----------
    n_iterations : int
        Sifting steps per IMF.
    """

    n_iterations: int = 10

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be at least 1, got {self.n_iterations}")

    def should_stop(self, state: SiftState) -> bool:
        """Accept the candidate once the iteration budget is spent."""
        return state.iteration >= self.n_iterations


def stopping_from_name(name: str, **parameters: float) -> StoppingCriterion:
    """Build a criterion from its configuration name.

    Parameters
    ----------
    name : {"three_threshold", "cauchy", "fixed"}
        Criterion name.
    **parameters
        Keyword arguments of the criterion's constructor.

    Returns
    -------
    StoppingCriterion
        The configured criterion.

    Examples
    --------
    >>> stopping_from_name("fixed", n_iterations=8)
    FixedIterationStopping(n_iterations=8)
    """
    if name == "three_threshold":
        return ThreeThresholdStopping(**parameters)
    if name == "cauchy":
        return CauchyStopping(**parameters)  # type: ignore[arg-type]
    if name == "fixed":
        return FixedIterationStopping(n_iterations=int(parameters.get("n_iterations", 10)))
    raise ValueError(f"unknown stopping criterion {name!r}; choose one of {STOPPING_CRITERIA}")
