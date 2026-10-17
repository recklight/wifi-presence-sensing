"""Putting reported CSI on an absolute scale.

The complex values in a beamforming report are 8-bit integers whose magnitude
follows the receiver's automatic gain control, not the physical channel.  When
the AGC steps by a few dB between packets, every reported amplitude steps with
it.  Fed straight into a time-series decomposition, those steps look like
genuine low-frequency modes.

The report also carries what is needed to undo this.  Each receive chain's
RSSI is given in dB relative to the AGC setting, so the total received power is

.. math::

    P_{rx}\\,[\\mathrm{dBm}] = 10\\log_{10}\\Big(\\sum_{c} 10^{\\mathrm{RSSI}_c/10}\\Big)
                              - 44 - \\mathrm{AGC},

where the constant 44 dB is the Intel 5300's RSSI offset.  Scaling the CSI so
that its mean power per subcarrier equals :math:`P_{rx}` removes the AGC from
the amplitudes:

.. math::

    s = \\frac{10^{P_{rx}/10}}{\\frac{1}{N_{sc}}\\sum_{t,r,k} |C_{t,r,k}|^2},
    \\qquad C_{abs} = \\sqrt{s}\\, C,

with :math:`C_{abs}` in :math:`\\sqrt{\\mathrm{mW}}`.  Dividing additionally by
the noise power (thermal plus the quantization noise of the 8-bit report)
gives CSI in SNR units, which :mod:`wifi_presence_sensing.calibrate.link_quality` needs.

When several transmit chains share the radio's power budget, each chain
radiates only a fraction of it; ``chain_power_correction_db`` restores the
per-stream SNR (3 dB for two chains; the Intel 5300 is commonly calibrated at
4.5 dB for three).

The RSSI offset, the quantization-noise term and the chain power correction
are those documented with the Linux 802.11n CSI Tool (Halperin et al., 2011).

References
----------
D. Halperin, W. Hu, A. Sheth, D. Wetherall, "Tool Release: Gathering 802.11n
Traces with Channel State Information," ACM SIGCOMM Computer Communication
Review, 41(1):53, 2011.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

_LOGGER = logging.getLogger(__name__)

#: Offset between the reported RSSI scale and dBm on the Intel 5300.
RSSI_OFFSET_DB = 44.0

#: Default per-stream power correction, keyed by the number of transmit chains.
DEFAULT_CHAIN_POWER_CORRECTION_DB: dict[int, float] = {
    1: 0.0,
    2: 10.0 * np.log10(2.0),
    3: 4.5,
}

#: Calibration references accepted by :func:`to_calibrated_csi`.
CALIBRATION_REFERENCES: tuple[str, ...] = ("absolute", "noise")


def to_db(power: np.ndarray | float) -> np.ndarray:
    """Linear power to decibels, ``10 log10(power)``.

    Examples
    --------
    >>> float(to_db(100.0))
    20.0
    """
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.asarray(power, dtype=np.float64))


def from_db(level_db: np.ndarray | float) -> np.ndarray:
    """Decibels to linear power, ``10 ** (dB / 10)``.

    Examples
    --------
    >>> float(from_db(30.0))
    1000.0
    """
    return np.power(10.0, np.asarray(level_db, dtype=np.float64) / 10.0)


def amplitude_db(csi: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """Amplitude in decibels, ``20 log10 |csi|``, with a floor against ``log(0)``.

    Examples
    --------
    >>> amplitude_db(np.array([10.0 + 0j, 1j])).tolist()
    [20.0, 0.0]
    """
    return 20.0 * np.log10(np.maximum(np.abs(csi), floor))


def total_received_power_dbm(rssi_db: np.ndarray, agc_db: np.ndarray | float) -> np.ndarray:
    """Total received power from per-chain RSSI and the AGC setting.

    Parameters
    ----------
    rssi_db : numpy.ndarray
        Per-chain RSSI, shaped ``(..., n_chains)``; zero marks an absent chain.
    agc_db : numpy.ndarray or float
        AGC setting, broadcastable to ``rssi_db.shape[:-1]``.

    Returns
    -------
    numpy.ndarray
        Power in dBm; NaN where no chain reported.

    Examples
    --------
    Two chains at the same level add 3 dB:

    >>> round(float(total_received_power_dbm(np.array([30, 30, 0]), 30)), 2)
    -40.99
    """
    rssi = np.asarray(rssi_db, dtype=np.float64)
    linear = np.where(rssi != 0, from_db(rssi), 0.0).sum(axis=-1)
    total = np.full(linear.shape, np.nan)
    present = linear > 0
    total[present] = to_db(linear[present])
    return total - RSSI_OFFSET_DB - np.asarray(agc_db, dtype=np.float64)


def chain_power_correction(n_tx: int, table_db: Mapping[int, float] | None = None) -> float:
    """Amplitude factor restoring per-stream power when ``n_tx`` chains share the budget.

    Examples
    --------
    >>> round(chain_power_correction(2), 6)
    1.414214
    """
    table = DEFAULT_CHAIN_POWER_CORRECTION_DB if table_db is None else table_db
    if n_tx not in table:
        raise ValueError(f"no chain power correction defined for n_tx={n_tx}")
    return float(np.sqrt(from_db(table[n_tx])))


def to_calibrated_csi(
    csi: np.ndarray,
    rssi_db: np.ndarray,
    agc_db: np.ndarray | float,
    *,
    noise_dbm: np.ndarray | float | None = None,
    reference: str = "absolute",
    include_quantization_noise: bool = True,
    chain_power_correction_db: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Scale reported CSI to an absolute or an SNR reference.

    Parameters
    ----------
    csi : numpy.ndarray
        Reported CSI, ``(n_tx, n_rx, n_sc)`` for one packet or
        ``(n_packets, n_tx, n_rx, n_sc)`` for a trace.
    rssi_db : numpy.ndarray
        Per-chain RSSI, ``(3,)`` or ``(n_packets, 3)``.
    agc_db : numpy.ndarray or float
        AGC setting, scalar or ``(n_packets,)``.
    noise_dbm : numpy.ndarray or float, optional
        Noise floor, required for ``reference="noise"``.  Pass the output of
        :class:`wifi_presence_sensing.calibrate.noise_model.NoiseModel`, not the raw field.
    reference : {"absolute", "noise"}, optional
        ``absolute``: :math:`|C|^2` in milliwatts.  ``noise``: :math:`|C|^2`
        in units of the noise power, i.e. per-entry SNR.
    include_quantization_noise : bool, optional
        Add the 8-bit rounding noise to the noise power (``noise`` reference).
    chain_power_correction_db : mapping, optional
        Overrides :data:`DEFAULT_CHAIN_POWER_CORRECTION_DB`.

    Returns
    -------
    numpy.ndarray
        Complex CSI with the same shape as ``csi``.

    Raises
    ------
    ValueError
        If the reference is unknown, the noise is missing, or shapes disagree.

    Examples
    --------
    The same channel reported at two AGC settings calibrates to the same values:

    >>> h = np.full((1, 3, 30), 20 + 0j)
    >>> a = to_calibrated_csi(h, np.array([30, 30, 30]), 30)
    >>> b = to_calibrated_csi(2 * h, np.array([36, 36, 36]), 36)
    >>> bool(np.allclose(a, b))
    True
    """
    if reference not in CALIBRATION_REFERENCES:
        raise ValueError(f"reference={reference!r} is not one of {CALIBRATION_REFERENCES}")
    tensor = np.asarray(csi, dtype=np.complex128)
    single = tensor.ndim == 3
    if single:
        tensor = tensor[None]
    if tensor.ndim != 4:
        raise ValueError(f"CSI must be 3-D or 4-D, got shape {np.shape(csi)}")
    n_packets, n_tx, n_rx, n_sc = tensor.shape

    rssi = np.asarray(rssi_db, dtype=np.float64).reshape(n_packets, -1)
    agc = np.broadcast_to(np.asarray(agc_db, dtype=np.float64), (n_packets,))
    received_mw = from_db(total_received_power_dbm(rssi, agc))
    csi_power = (np.abs(tensor) ** 2).sum(axis=(1, 2, 3)) / n_sc
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(csi_power > 0, received_mw / csi_power, np.nan)
    if np.isnan(scale).any():
        _LOGGER.warning(
            "%d packet(s) have no RSSI or an all-zero CSI and calibrate to NaN",
            int(np.isnan(scale).sum()),
        )

    if reference == "absolute":
        factor = np.sqrt(scale)
    else:
        if noise_dbm is None:
            raise ValueError("reference='noise' needs the noise floor (noise_dbm)")
        noise_mw = from_db(np.broadcast_to(np.asarray(noise_dbm, dtype=np.float64), (n_packets,)))
        if include_quantization_noise:
            # Rounding each I/Q value to an integer adds about unit variance per
            # entry in report units; scale converts that to milliwatts.
            noise_mw = noise_mw + scale * n_tx * n_rx
        factor = np.sqrt(scale / noise_mw) * chain_power_correction(n_tx, chain_power_correction_db)

    calibrated = tensor * factor[:, None, None, None]
    return calibrated[0] if single else calibrated


def detrended_phase(csi: np.ndarray, subcarrier_indices: np.ndarray) -> np.ndarray:
    """Unwrapped phase across subcarriers with its linear trend removed.

    A residual carrier- and sampling-frequency offset adds a phase term that is
    affine in the subcarrier index and changes from packet to packet.
    Removing the least-squares line across subcarriers (slope and offset)
    leaves the part of the phase that reflects the propagation channel.

    Parameters
    ----------
    csi : numpy.ndarray
        Complex CSI with subcarriers on the last axis.
    subcarrier_indices : numpy.ndarray
        OFDM index of each subcarrier, e.g. from
        :func:`wifi_presence_sensing.capture.subcarriers.reported_subcarrier_indices`.

    Returns
    -------
    numpy.ndarray
        Phase in radians, same shape as ``csi``.

    Examples
    --------
    >>> k = np.arange(-28, 29, 2)
    >>> h = np.exp(1j * (0.3 * k + 1.0))
    >>> bool(np.allclose(detrended_phase(h, k), 0.0))
    True
    """
    indices = np.asarray(subcarrier_indices, dtype=np.float64)
    tensor = np.asarray(csi)
    if tensor.shape[-1] != indices.size:
        raise ValueError(f"{indices.size} subcarrier indices for {tensor.shape[-1]} subcarriers")
    phase = np.unwrap(np.angle(tensor), axis=-1)
    centred = indices - indices.mean()
    slope = (phase * centred).sum(axis=-1, keepdims=True) / (centred**2).sum()
    offset = phase.mean(axis=-1, keepdims=True)
    return phase - offset - slope * centred
