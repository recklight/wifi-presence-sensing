"""Immutable container for one beamforming report.

A :class:`ChannelSnapshot` always holds a three-dimensional CSI tensor shaped
``(n_tx, n_rx, n_subcarriers)``, including single-transmit-chain reports.  A
report with one transmit chain is ``(1, n_rx, 30)``, never squeezed to
``(n_rx, 30)``: a tensor whose rank depends on the packet is how indexing bugs
creep into per-antenna-pair features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rate_flags import RateInfo

#: Width of the NIC timestamp counter, in microseconds.
TIMESTAMP_PERIOD_US = 2**32


@dataclass(frozen=True)
class RadioMeta:
    """Radio-level metadata of one beamforming report.

    Attributes
    ----------
    report_counter : int
        Running report counter maintained by the driver.
    n_tx, n_rx : int
        Number of transmit and receive chains in the CSI tensor.
    rssi_db : tuple of int
        Per-receive-chain RSSI ``(a, b, c)`` in dB relative to the AGC
        setting; 0 marks an absent chain.
    noise_dbm : int
        Noise floor reported by the NIC, in dBm.  Known to be unreliable;
        see :mod:`wifi_presence_sensing.calibrate.noise_model`.
    agc_db : int
        Automatic gain control setting, in dB.
    antenna_permutation : tuple of int
        For each reported receive column ``i``, the 0-based physical antenna
        it came from.
    permutation_valid : bool
        False when the permutation field was not a valid permutation; the
        tensor is then left in reported order.
    rate : RateInfo
        Decoded rate-and-flags word.
    """

    report_counter: int
    n_tx: int
    n_rx: int
    rssi_db: tuple[int, int, int]
    noise_dbm: int
    agc_db: int
    antenna_permutation: tuple[int, ...]
    permutation_valid: bool
    rate: RateInfo

    @property
    def bandwidth_mhz(self) -> int:
        """Channel bandwidth of the frame, 20 or 40 MHz."""
        return self.rate.bandwidth_mhz


@dataclass(frozen=True, eq=False)
class ChannelSnapshot:
    """One beamforming report: a CSI tensor, its timestamp and its metadata.

    Parameters
    ----------
    csi : numpy.ndarray
        Complex array shaped ``(n_tx, n_rx, n_subcarriers)``.  Stored as a
        read-only ``complex128`` copy.
    timestamp_us : int
        Raw 32-bit NIC timestamp in microseconds.  It wraps every
        :data:`TIMESTAMP_PERIOD_US`; use
        :func:`wifi_presence_sensing.capture.timeline.unwrap_timestamps` on a sequence.
    meta : RadioMeta
        Radio metadata.

    Raises
    ------
    ValueError
        If the tensor shape disagrees with ``meta`` or the timestamp is out of
        the 32-bit range.
    """

    csi: np.ndarray
    timestamp_us: int
    meta: RadioMeta

    def __post_init__(self) -> None:
        csi = np.array(self.csi, dtype=np.complex128, copy=True)
        if csi.ndim != 3:
            raise ValueError(
                f"CSI must be shaped (n_tx, n_rx, n_subcarriers); got {csi.ndim} dimensions"
            )
        if csi.shape[:2] != (self.meta.n_tx, self.meta.n_rx):
            raise ValueError(
                f"CSI shape {csi.shape} disagrees with n_tx={self.meta.n_tx}, n_rx={self.meta.n_rx}"
            )
        if not 0 <= int(self.timestamp_us) < TIMESTAMP_PERIOD_US:
            raise ValueError(f"timestamp {self.timestamp_us} is outside the 32-bit counter range")
        csi.setflags(write=False)
        object.__setattr__(self, "csi", csi)
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))

    @property
    def radio_setup(self) -> tuple[int, int, int]:
        """``(bandwidth_mhz, n_tx, n_rx)``: the key that decides stackability."""
        return (self.meta.bandwidth_mhz, self.meta.n_tx, self.meta.n_rx)
