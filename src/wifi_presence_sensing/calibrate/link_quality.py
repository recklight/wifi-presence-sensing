"""Bit error rates and the effective SNR of a frequency-selective MIMO channel.

The arithmetic mean of per-subcarrier SNRs is a poor predictor of how a link
performs: one deep fade costs far more bit errors than a strong subcarrier
saves.  The effective SNR therefore averages in the bit-error-rate domain and
maps the mean back to an SNR (Halperin et al., 2010):

.. math::

    \\gamma_{\\mathrm{eff}} = \\mathrm{BER}_m^{-1}\\Big(\\frac{1}{K}
        \\sum_{k=1}^{K} \\mathrm{BER}_m(\\gamma_k)\\Big),

where :math:`\\gamma_k` runs over every subcarrier of every spatial stream and
:math:`m` is the modulation.  The uncoded AWGN bit error rates, with
:math:`\\gamma` the linear SNR per symbol and :math:`Q` the Gaussian tail
function, are

========  =====================================================
BPSK      :math:`Q(\\sqrt{2\\gamma}) = \\tfrac12\\,\\mathrm{erfc}(\\sqrt{\\gamma})`
QPSK      :math:`Q(\\sqrt{\\gamma})`
16-QAM    :math:`\\tfrac34\\,Q(\\sqrt{\\gamma/5})`
64-QAM    :math:`\\tfrac{7}{12}\\,Q(\\sqrt{\\gamma/21})`
========  =====================================================

(the square-QAM entries are the usual nearest-neighbour Gray-coding
approximations, Proakis & Salehi).  Each has the form :math:`c\\,Q(\\sqrt{\\gamma/d})`
and the closed-form inverse :math:`\\gamma = d\\,[Q^{-1}(p/c)]^2`.

Numerics.  At high SNR the bit error rate underflows double precision (BPSK at
30 dB is about :math:`10^{-436}`), after which the inverse returns infinity.
:class:`ModulationModel` therefore also works with the *logarithm* of the
error rate through :func:`scipy.special.log_ndtr` and its inverse, and
:func:`effective_snr` averages in that domain with a log-sum-exp.  The result
is exact across the whole SNR range, and a flat channel round-trips to its own
SNR.

References
----------
D. Halperin, W. Hu, A. Sheth, D. Wetherall, "Predictable 802.11 Packet
Delivery from Wireless Channel Measurements," ACM SIGCOMM, 2010.

J. G. Proakis, M. Salehi, Digital Communications, 5th ed., McGraw-Hill, 2008.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.special import logsumexp, ndtri_exp

#: Smallest bit error rate the linear-domain inverse accepts before saturating.
MIN_ERROR_RATE = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True)
class ModulationModel:
    """Uncoded AWGN bit error rate of the form :math:`c\\,Q(\\sqrt{\\gamma/d})`.

    Parameters
    ----------
    name : str
        Modulation name.
    bits_per_symbol : int
        Bits carried per constellation symbol.
    coefficient : float
        :math:`c`, the average number of bit errors per nearest-neighbour
        symbol error.
    snr_divisor : float
        :math:`d`.

    Examples
    --------
    >>> qpsk = MODULATION_MODELS["qpsk"]
    >>> f"{float(qpsk.error_rate(10.0)):.4e}"
    '7.8270e-04'
    >>> round(float(qpsk.snr_from_error_rate(qpsk.error_rate(10.0))), 10)
    10.0
    """

    name: str
    bits_per_symbol: int
    coefficient: float
    snr_divisor: float

    @property
    def max_error_rate(self) -> float:
        """Error rate at zero SNR, :math:`c/2`."""
        return 0.5 * self.coefficient

    def log_error_rate(self, snr_linear: np.ndarray | float) -> np.ndarray:
        """Natural log of the bit error rate; exact even where the rate underflows."""
        snr = _check_snr(snr_linear)
        return np.log(self.coefficient) + special.log_ndtr(-np.sqrt(snr / self.snr_divisor))

    def error_rate(self, snr_linear: np.ndarray | float) -> np.ndarray:
        """Bit error rate at a linear per-symbol SNR."""
        return np.exp(self.log_error_rate(snr_linear))

    def snr_from_log_error_rate(self, log_error_rate: np.ndarray | float) -> np.ndarray:
        """Inverse of :meth:`log_error_rate`.

        Log error rates at or above ``log(c/2)`` (no better than guessing)
        map to zero SNR.
        """
        log_p = np.asarray(log_error_rate, dtype=np.float64) - np.log(self.coefficient)
        tail = np.minimum(log_p, np.log(0.5))
        argument = -ndtri_exp(tail)  # Q^{-1}(p) = -Phi^{-1}(p)
        return self.snr_divisor * np.maximum(argument, 0.0) ** 2

    def snr_from_error_rate(self, error_rate: np.ndarray | float) -> np.ndarray:
        """Inverse of :meth:`error_rate`, with the input clipped to ``(0, c/2]``."""
        p = np.asarray(error_rate, dtype=np.float64)
        if np.any(np.isnan(p)) or np.any(p < 0):
            raise ValueError("bit error rates must be non-negative numbers")
        clipped = np.clip(p, MIN_ERROR_RATE, self.max_error_rate)
        return self.snr_from_log_error_rate(np.log(clipped))


def _check_snr(snr_linear: np.ndarray | float) -> np.ndarray:
    snr = np.asarray(snr_linear, dtype=np.float64)
    if np.any(np.isnan(snr)) or np.any(snr < 0):
        raise ValueError("SNR must be given as non-negative linear values (not dB)")
    return snr


#: Registered modulations, keyed by name.
MODULATION_MODELS: dict[str, ModulationModel] = {
    "bpsk": ModulationModel("bpsk", 1, 1.0, 0.5),
    "qpsk": ModulationModel("qpsk", 2, 1.0, 1.0),
    "qam16": ModulationModel("qam16", 4, 0.75, 5.0),
    "qam64": ModulationModel("qam64", 6, 7.0 / 12.0, 21.0),
}


def _model(modulation: str) -> ModulationModel:
    try:
        return MODULATION_MODELS[modulation]
    except KeyError:
        raise ValueError(
            f"unknown modulation {modulation!r}; choose one of {sorted(MODULATION_MODELS)}"
        ) from None


def bpsk_error_rate(snr_linear: np.ndarray | float) -> np.ndarray:
    """BPSK bit error rate, :math:`\\tfrac12\\mathrm{erfc}(\\sqrt{\\gamma}) = Q(\\sqrt{2\\gamma})`.

    Examples
    --------
    >>> round(float(bpsk_error_rate(1.0)), 6)
    0.07865
    """
    return 0.5 * special.erfc(np.sqrt(_check_snr(snr_linear)))


def bpsk_error_rate_inverse(error_rate: np.ndarray | float) -> np.ndarray:
    """SNR at which BPSK reaches a bit error rate, :math:`[\\mathrm{erfc}^{-1}(2p)]^2`.

    The rate is clipped to ``[tiny, 0.5]``, so an underflowed zero saturates at
    about 704 (28.5 dB) instead of returning infinity; use
    ``MODULATION_MODELS["bpsk"].snr_from_log_error_rate`` beyond that.

    Examples
    --------
    >>> round(float(bpsk_error_rate_inverse(bpsk_error_rate(4.0))), 10)
    4.0
    """
    p = np.clip(np.asarray(error_rate, dtype=np.float64), MIN_ERROR_RATE, 0.5)
    return special.erfcinv(2.0 * p) ** 2


def qpsk_error_rate(snr_linear: np.ndarray | float) -> np.ndarray:
    """QPSK bit error rate, :math:`Q(\\sqrt{\\gamma})`."""
    return MODULATION_MODELS["qpsk"].error_rate(snr_linear)


def qpsk_error_rate_inverse(error_rate: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`qpsk_error_rate`."""
    return MODULATION_MODELS["qpsk"].snr_from_error_rate(error_rate)


def qam16_error_rate(snr_linear: np.ndarray | float) -> np.ndarray:
    """16-QAM bit error rate, :math:`\\tfrac34 Q(\\sqrt{\\gamma/5})`."""
    return MODULATION_MODELS["qam16"].error_rate(snr_linear)


def qam16_error_rate_inverse(error_rate: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`qam16_error_rate`."""
    return MODULATION_MODELS["qam16"].snr_from_error_rate(error_rate)


def qam64_error_rate(snr_linear: np.ndarray | float) -> np.ndarray:
    """64-QAM bit error rate, :math:`\\tfrac{7}{12} Q(\\sqrt{\\gamma/21})`."""
    return MODULATION_MODELS["qam64"].error_rate(snr_linear)


def qam64_error_rate_inverse(error_rate: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`qam64_error_rate`."""
    return MODULATION_MODELS["qam64"].snr_from_error_rate(error_rate)


def effective_snr(
    snr_linear: np.ndarray,
    modulation: str = "qpsk",
    *,
    axis: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Effective SNR: invert the average bit error rate over subcarriers and streams.

    Parameters
    ----------
    snr_linear : numpy.ndarray
        Linear per-symbol SNR of every subcarrier (and stream).
    modulation : str, optional
        A key of :data:`MODULATION_MODELS`.
    axis : int or sequence of int, optional
        Axes to average over; ``None`` averages everything.

    Returns
    -------
    numpy.ndarray
        Linear effective SNR.

    Examples
    --------
    A flat channel keeps its SNR; a single deep fade pulls the effective SNR
    far below the arithmetic mean:

    >>> round(float(effective_snr(np.full(30, 100.0), "qam16")), 8)
    100.0
    >>> faded = np.r_[np.full(29, 100.0), 1.0]
    >>> bool(effective_snr(faded, "qpsk") < 0.5 * faded.mean())
    True
    """
    model = _model(modulation)
    log_ber = model.log_error_rate(snr_linear)
    if axis is None:
        count = log_ber.size
        mean_log = logsumexp(log_ber) - np.log(count)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = int(np.prod([log_ber.shape[a] for a in axes]))
        mean_log = logsumexp(log_ber, axis=axes) - np.log(count)
    return model.snr_from_log_error_rate(mean_log)


def mmse_stream_snr(channel: np.ndarray) -> np.ndarray:
    """Per-stream post-detection SNR of a linear MMSE receiver.

    With the channel in SNR units (noise power 1, see
    :func:`wifi_presence_sensing.calibrate.scaling.to_calibrated_csi`), stream :math:`s`
    on a subcarrier with channel matrix :math:`H` reaches

    .. math::

        \\gamma_s = \\frac{1}{[(I + H^{H} H)^{-1}]_{ss}} - 1.

    Parameters
    ----------
    channel : numpy.ndarray
        Transmit-major CSI in SNR units, ``(..., n_tx, n_rx, n_subcarriers)``.

    Returns
    -------
    numpy.ndarray
        Linear SNR shaped ``(..., n_subcarriers, n_tx)``.

    Examples
    --------
    A single stream reduces to maximum-ratio combining, :math:`\\|h\\|^2`:

    >>> h = np.ones((1, 3, 30)) * 2.0
    >>> bool(np.allclose(mmse_stream_snr(h), 12.0))
    True
    """
    tensor = np.asarray(channel, dtype=np.complex128)
    if tensor.ndim < 3:
        raise ValueError("channel must be shaped (..., n_tx, n_rx, n_subcarriers)")
    n_tx = tensor.shape[-3]
    gram = np.einsum("...ark,...brk->...kab", tensor.conj(), tensor)
    inverse = np.linalg.inv(gram + np.eye(n_tx))
    diagonal = np.real(np.diagonal(inverse, axis1=-2, axis2=-1))
    return np.maximum(1.0 / diagonal - 1.0, 0.0)


def packet_effective_snr_db(
    channel: np.ndarray, modulations: Sequence[str] = ("bpsk", "qpsk", "qam16", "qam64")
) -> dict[str, np.ndarray]:
    """Effective SNR in dB of each packet for several modulations.

    Parameters
    ----------
    channel : numpy.ndarray
        CSI in SNR units, ``(n_packets, n_tx, n_rx, n_subcarriers)``.
    modulations : sequence of str, optional
        Keys of :data:`MODULATION_MODELS`.

    Returns
    -------
    dict
        ``{modulation: effective SNR in dB per packet}``.
    """
    tensor = np.asarray(channel)
    if tensor.ndim != 4:
        raise ValueError("channel must be shaped (n_packets, n_tx, n_rx, n_subcarriers)")
    stream_snr = mmse_stream_snr(tensor)  # (n_packets, n_sc, n_tx)
    result: dict[str, np.ndarray] = {}
    for name in modulations:
        linear = effective_snr(stream_snr, name, axis=(1, 2))
        with np.errstate(divide="ignore"):
            result[name] = 10.0 * np.log10(linear)
    return result
