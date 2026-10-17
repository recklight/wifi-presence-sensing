"""Stacking snapshots into a fixed-shape trace.

A capture is not guaranteed to be homogeneous: the number of transmit chains
can change from packet to packet, and in principle so can the bandwidth.
Stacked as-is, such a sequence either fails or gives an array whose "antenna
pair 3" means different things in different rows.

The triple ``(bandwidth_mhz, n_tx, n_rx)`` of a report is its *radio setup*.
:func:`stack_snapshots` never concatenates across radio setups.  Bandwidth is
always resolved to the dominant one (the minority is dropped and reported).
For the transmit chains there are two policies:

``"dominant"``
    Keep only packets with the most common ``n_tx``.
``"first_chain"``
    Keep every packet with the dominant bandwidth and ``n_rx``, but retain
    only transmit chain 0, so single- and multi-chain packets share a
    ``(1, n_rx, 30)`` shape and no packet is lost.  Two caveats: a packet
    sent on several chains splits its power among them, so after absolute
    scaling the level of chain 0 is only approximately comparable between
    single- and multi-chain packets; and chain 0 of a spatially mapped report
    still mixes every chain, which is why the configuration rejects this
    policy together with a spatial mapping.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .snapshot import ChannelSnapshot
from .timeline import unwrap_timestamps

_LOGGER = logging.getLogger(__name__)

#: Accepted values of the ``tx_policy`` argument.
TX_POLICIES: tuple[str, ...] = ("dominant", "first_chain")


@dataclass(frozen=True, eq=False)
class ChannelTrace:
    """A homogeneous sequence of CSI reports stacked into arrays.

    Attributes
    ----------
    csi : numpy.ndarray
        Complex array shaped ``(n_packets, n_tx, n_rx, n_subcarriers)``.
    timestamps_s : numpy.ndarray
        Unwrapped packet times in seconds, relative to the first packet of the
        original capture.
    rssi_db : numpy.ndarray
        Per-chain RSSI, shaped ``(n_packets, 3)``.
    noise_dbm : numpy.ndarray
        Reported noise floor per packet.
    agc_db : numpy.ndarray
        AGC setting per packet.
    bandwidth_mhz : int
        Channel bandwidth shared by every packet.
    source : str
        Where the trace came from (file name or a label).
    """

    csi: np.ndarray
    timestamps_s: np.ndarray
    rssi_db: np.ndarray
    noise_dbm: np.ndarray
    agc_db: np.ndarray
    bandwidth_mhz: int
    source: str = ""

    def __post_init__(self) -> None:
        n = self.csi.shape[0]
        if self.csi.ndim != 4:
            raise ValueError(f"trace CSI must be 4-D, got shape {self.csi.shape}")
        for name in ("timestamps_s", "noise_dbm", "agc_db"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one entry per packet ({n})")
        if self.rssi_db.shape != (n, 3):
            raise ValueError(f"rssi_db must be shaped ({n}, 3), got {self.rssi_db.shape}")

    @property
    def n_packets(self) -> int:
        """Number of packets."""
        return int(self.csi.shape[0])

    @property
    def n_tx(self) -> int:
        """Number of transmit chains."""
        return int(self.csi.shape[1])

    @property
    def n_rx(self) -> int:
        """Number of receive chains."""
        return int(self.csi.shape[2])

    def select(self, mask: np.ndarray) -> ChannelTrace:
        """Return the packets selected by a boolean mask or an index array."""
        return ChannelTrace(
            csi=self.csi[mask],
            timestamps_s=self.timestamps_s[mask],
            rssi_db=self.rssi_db[mask],
            noise_dbm=self.noise_dbm[mask],
            agc_db=self.agc_db[mask],
            bandwidth_mhz=self.bandwidth_mhz,
            source=self.source,
        )


@dataclass(frozen=True)
class StackingReport:
    """What :func:`stack_snapshots` kept and dropped.

    Attributes
    ----------
    n_input : int
        Snapshots offered.
    n_kept : int
        Snapshots in the trace.
    radio_setups : dict
        Count of each ``"<bw>MHz/<n_tx>x<n_rx>"`` radio setup in the input.
    dropped_bandwidth : int
        Packets dropped because their bandwidth was not the dominant one.
    dropped_chain_mismatch : int
        Packets dropped because their chain counts did not match.
    tx_policy : str
        Policy applied.
    """

    n_input: int
    n_kept: int
    radio_setups: dict[str, int] = field(default_factory=dict)
    dropped_bandwidth: int = 0
    dropped_chain_mismatch: int = 0
    tx_policy: str = "dominant"


def radio_setup_counts(snapshots: Sequence[ChannelSnapshot]) -> Counter[tuple[int, int, int]]:
    """Count the ``(bandwidth_mhz, n_tx, n_rx)`` radio setups of a capture."""
    return Counter(s.radio_setup for s in snapshots)


def stack_snapshots(
    snapshots: Sequence[ChannelSnapshot], *, tx_policy: str = "dominant", source: str = ""
) -> tuple[ChannelTrace, StackingReport]:
    """Stack snapshots into a homogeneous :class:`ChannelTrace`.

    Parameters
    ----------
    snapshots : sequence of ChannelSnapshot
        Reports in arrival order.
    tx_policy : {"dominant", "first_chain"}, optional
        How to reconcile packets with different numbers of transmit chains.
    source : str, optional
        Label stored on the trace.

    Returns
    -------
    trace : ChannelTrace
        The stacked trace.
    report : StackingReport
        What was kept and dropped.

    Raises
    ------
    ValueError
        If no snapshot is given, the policy is unknown, or nothing survives.
    """
    if tx_policy not in TX_POLICIES:
        raise ValueError(f"tx_policy={tx_policy!r} is not one of {TX_POLICIES}")
    if not snapshots:
        raise ValueError("cannot stack an empty capture")

    counts = radio_setup_counts(snapshots)
    # Unwrap on the full sequence so wraps hidden behind dropped packets still count.
    unwrapped = unwrap_timestamps(np.array([s.timestamp_us for s in snapshots], dtype=np.int64))
    times = (unwrapped - unwrapped[0]).astype(np.float64) * 1e-6

    bandwidths = Counter(s.meta.bandwidth_mhz for s in snapshots)
    bandwidth = bandwidths.most_common(1)[0][0]
    same_bandwidth = np.array([s.meta.bandwidth_mhz == bandwidth for s in snapshots])
    dropped_bandwidth = int(np.count_nonzero(~same_bandwidth))
    if dropped_bandwidth:
        _LOGGER.warning(
            "%s: %d packet(s) at another bandwidth than the dominant %d MHz were dropped",
            source or "trace",
            dropped_bandwidth,
            bandwidth,
        )

    rx_counts = Counter(
        s.meta.n_rx for s, same in zip(snapshots, same_bandwidth, strict=True) if same
    )
    n_rx = rx_counts.most_common(1)[0][0]
    if tx_policy == "dominant":
        tx_counts = Counter(
            s.meta.n_tx
            for s, same in zip(snapshots, same_bandwidth, strict=True)
            if same and s.meta.n_rx == n_rx
        )
        n_tx = tx_counts.most_common(1)[0][0]
        keep = same_bandwidth & np.array(
            [s.meta.n_tx == n_tx and s.meta.n_rx == n_rx for s in snapshots]
        )
        select_chain = slice(None)
    else:
        keep = same_bandwidth & np.array([s.meta.n_rx == n_rx for s in snapshots])
        select_chain = slice(0, 1)

    dropped_chain_mismatch = int(np.count_nonzero(same_bandwidth & ~keep))
    if dropped_chain_mismatch:
        _LOGGER.info(
            "%s: %d packet(s) with a minority chain count were dropped (policy %r)",
            source or "trace",
            dropped_chain_mismatch,
            tx_policy,
        )
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError("no packet survived the radio-setup filtering")

    kept = [snapshots[i] for i in indices]
    trace = ChannelTrace(
        csi=np.stack([s.csi[select_chain] for s in kept]),
        timestamps_s=times[indices],
        rssi_db=np.array([s.meta.rssi_db for s in kept], dtype=np.float64),
        noise_dbm=np.array([s.meta.noise_dbm for s in kept], dtype=np.float64),
        agc_db=np.array([s.meta.agc_db for s in kept], dtype=np.float64),
        bandwidth_mhz=int(bandwidth),
        source=source,
    )
    report = StackingReport(
        n_input=len(snapshots),
        n_kept=int(indices.size),
        radio_setups={f"{bw}MHz/{tx}x{rx}": n for (bw, tx, rx), n in sorted(counts.items())},
        dropped_bandwidth=dropped_bandwidth,
        dropped_chain_mismatch=dropped_chain_mismatch,
        tx_policy=tx_policy,
    )
    return trace, report
