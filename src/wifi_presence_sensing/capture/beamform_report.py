"""Reader and writer for 802.11n beamforming-report capture files.

File format
-----------
A capture is a flat stream of frames.  Each frame starts with a big-endian
``uint16`` that counts the bytes that follow it (one code byte plus the
payload), then the code byte.  Code ``0xBB`` marks a beamforming report; other
codes are skipped.  A beamforming payload is a 20-byte little-endian header

=======  =====  ========================================================
offset   type   field
=======  =====  ========================================================
0        u32    NIC timestamp, microseconds, wraps at 2**32
4        u16    report counter
6        --     two reserved bytes
8        u8     number of receive chains ``n_rx``
9        u8     number of transmit chains ``n_tx``
10-12    u8     RSSI of receive chains a, b, c (dB relative to AGC)
13       i8     noise floor, dBm
14       u8     AGC setting, dB
15       u8     antenna selection (three 2-bit fields)
16       u16    length of the packed CSI block in bytes
18       u16    rate and flags (MCS, bandwidth, guard interval)
=======  =====  ========================================================

followed by the bit-packed CSI.  For each of the 30 subcarrier groups the
packer skips three bits, then writes ``n_tx * n_rx`` complex entries as an
8-bit signed real part followed by an 8-bit signed imaginary part, LSB first.
Entries are ordered with the transmit chain varying fastest.  Because of the
three-bit skips, most values straddle a byte boundary.

The format is the one produced by the Linux 802.11n CSI Tool (Halperin et al.,
2011).  The bit offsets of every I/Q value depend only on ``(n_tx, n_rx)``, so
they are computed once and cached, and a whole block decodes with a few array
operations.

References
----------
D. Halperin, W. Hu, A. Sheth, D. Wetherall, "Tool Release: Gathering 802.11n
Traces with Channel State Information," ACM SIGCOMM Computer Communication
Review, 41(1):53, 2011.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .rate_flags import decode_rate_flags
from .snapshot import ChannelSnapshot, RadioMeta
from .subcarriers import N_SUBCARRIER_GROUPS

_LOGGER = logging.getLogger(__name__)

#: Frame code of a beamforming report.
BEAMFORM_REPORT_CODE = 0xBB

_HEADER = struct.Struct("<IH2xBBBBBbBBHH")
_FRAME_PREFIX = struct.Struct(">HB")

#: Size of the fixed payload header in bytes.
HEADER_SIZE = _HEADER.size

#: Bits skipped before each subcarrier group in the packed CSI block.
_GROUP_PAD_BITS = 3
#: Bits per complex entry (8-bit real + 8-bit imaginary).
_ENTRY_BITS = 16


class BeamformFormatError(ValueError):
    """Raised when a beamforming payload is malformed or internally inconsistent."""


@dataclass
class ReadStatistics:
    """Counters collected while reading one capture file.

    Attributes
    ----------
    n_frames : int
        Frames encountered, whatever their code.
    n_reports : int
        Beamforming reports successfully decoded.
    n_other_codes : int
        Frames with a code other than ``0xBB``, skipped.
    n_truncated : int
        Frames cut off by the end of the file.
    n_malformed : int
        Reports whose CSI length disagrees with ``n_tx``/``n_rx`` or whose
        payload is too short, skipped.
    n_invalid_permutation : int
        Reports whose antenna selection is not a permutation; kept in
        reported order.
    """

    n_frames: int = 0
    n_reports: int = 0
    n_other_codes: int = 0
    n_truncated: int = 0
    n_malformed: int = 0
    n_invalid_permutation: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dictionary."""
        return asdict(self)


def expected_csi_length(n_tx: int, n_rx: int) -> int:
    """Number of bytes the packed CSI block must occupy.

    Parameters
    ----------
    n_tx, n_rx : int
        Number of transmit and receive chains.

    Returns
    -------
    int
        ``ceil(30 * (16 * n_tx * n_rx + 3) / 8)``.

    Examples
    --------
    >>> expected_csi_length(2, 3)
    372
    >>> expected_csi_length(1, 3)
    192
    """
    bits = N_SUBCARRIER_GROUPS * (_ENTRY_BITS * n_tx * n_rx + _GROUP_PAD_BITS)
    return (bits + 7) // 8


@lru_cache(maxsize=32)
def _iq_bit_offsets(n_entries: int) -> np.ndarray:
    """Start bit of every real and imaginary byte, shaped ``(30, n_entries, 2)``.

    The real part of entry ``j`` in group ``g`` starts at
    ``3 * (g + 1) + 16 * (g * n_entries + j)``; the imaginary part follows
    eight bits later.
    """
    group = np.arange(N_SUBCARRIER_GROUPS, dtype=np.int64)[:, None]
    entry = np.arange(n_entries, dtype=np.int64)[None, :]
    real_start = _GROUP_PAD_BITS * (group + 1) + _ENTRY_BITS * (group * n_entries + entry)
    offsets = np.stack([real_start, real_start + 8], axis=-1)
    offsets.setflags(write=False)
    return offsets


def unpack_csi(block: bytes | np.ndarray, n_tx: int, n_rx: int) -> np.ndarray:
    """Decode a packed CSI block into a complex tensor in reported order.

    Parameters
    ----------
    block : bytes or numpy.ndarray
        The packed CSI bytes (at least :func:`expected_csi_length` long).
    n_tx, n_rx : int
        Number of transmit and receive chains.

    Returns
    -------
    numpy.ndarray
        Complex128 array shaped ``(n_tx, n_rx, 30)`` with integer-valued
        real and imaginary parts in ``[-128, 127]``.

    Raises
    ------
    BeamformFormatError
        If the block is shorter than the layout requires.
    """
    if n_tx < 1 or n_rx < 1:
        raise BeamformFormatError(f"n_tx and n_rx must be positive, got {n_tx}, {n_rx}")
    data = np.frombuffer(bytes(block), dtype=np.uint8)
    needed = expected_csi_length(n_tx, n_rx)
    if data.size < needed:
        raise BeamformFormatError(f"CSI block holds {data.size} bytes, layout needs {needed}")

    offsets = _iq_bit_offsets(n_tx * n_rx)
    byte_index = offsets >> 3
    shift = (offsets & 7).astype(np.uint16)
    # One zero byte of padding so the "next byte" lookup never runs off the end.
    padded = np.concatenate([data[:needed], np.zeros(1, dtype=np.uint8)]).astype(np.uint16)
    combined = (padded[byte_index] >> shift) | (padded[byte_index + 1] << (8 - shift))
    values = (combined & 0xFF).astype(np.uint8).view(np.int8).astype(np.float64)
    iq = values[..., 0] + 1j * values[..., 1]  # (30, n_entries), tx varies fastest
    return iq.reshape(N_SUBCARRIER_GROUPS, n_rx, n_tx).transpose(2, 1, 0).copy()


def pack_csi(csi: np.ndarray) -> bytes:
    """Encode a complex tensor into a packed CSI block; inverse of :func:`unpack_csi`.

    Parameters
    ----------
    csi : numpy.ndarray
        Complex array shaped ``(n_tx, n_rx, 30)`` in reported order, whose real
        and imaginary parts are integers in ``[-128, 127]``.

    Returns
    -------
    bytes
        The packed block, :func:`expected_csi_length` bytes long.

    Raises
    ------
    BeamformFormatError
        If the shape is wrong or a component is not an 8-bit signed integer.
    """
    tensor = np.asarray(csi)
    if tensor.ndim != 3 or tensor.shape[2] != N_SUBCARRIER_GROUPS:
        raise BeamformFormatError(f"CSI must be shaped (n_tx, n_rx, 30), got {tensor.shape}")
    n_tx, n_rx = tensor.shape[:2]
    iq = tensor.transpose(2, 1, 0).reshape(N_SUBCARRIER_GROUPS, n_rx * n_tx)
    parts = np.stack([iq.real, iq.imag], axis=-1)
    rounded = np.rint(parts)
    if not np.allclose(parts, rounded) or rounded.min() < -128 or rounded.max() > 127:
        raise BeamformFormatError(
            "CSI components must be integers in [-128, 127] to fit the 8-bit packed format"
        )
    raw = rounded.astype(np.int8).view(np.uint8)  # (30, n_entries, 2)

    offsets = _iq_bit_offsets(n_tx * n_rx)
    n_bytes = expected_csi_length(n_tx, n_rx)
    bit_positions = offsets[..., None] + np.arange(8)
    bit_values = (raw[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    bits = np.zeros(n_bytes * 8, dtype=np.uint8)
    bits[bit_positions.ravel()] = bit_values.ravel()
    return np.packbits(bits, bitorder="little").tobytes()


def decode_antenna_permutation(antenna_selection: int, n_rx: int) -> tuple[tuple[int, ...], bool]:
    """Decode the antenna-selection byte into a 0-based permutation.

    Parameters
    ----------
    antenna_selection : int
        The byte from the report header: three 2-bit fields, one per
        reported receive column.
    n_rx : int
        Number of receive chains actually reported.

    Returns
    -------
    permutation : tuple of int
        For each reported column, the 0-based physical antenna index.
    valid : bool
        True when the first ``n_rx`` fields form a permutation of
        ``range(n_rx)``.

    Examples
    --------
    >>> decode_antenna_permutation(0b100100, 3)   # fields 0, 1, 2
    ((0, 1, 2), True)
    >>> decode_antenna_permutation(0b000110, 3)   # fields 2, 1, 0
    ((2, 1, 0), True)
    """
    fields = tuple(int((antenna_selection >> (2 * i)) & 0b11) for i in range(n_rx))
    valid = sorted(fields) == list(range(n_rx))
    return fields, valid


def encode_antenna_selection(permutation: Iterable[int]) -> int:
    """Pack a 0-based permutation into the antenna-selection byte.

    Examples
    --------
    >>> encode_antenna_selection((2, 0, 1))
    18
    >>> decode_antenna_permutation(18, 3)
    ((2, 0, 1), True)
    """
    value = 0
    for i, antenna in enumerate(permutation):
        if not 0 <= int(antenna) <= 3:
            raise ValueError(f"antenna index must be in 0..3, got {antenna}")
        value |= int(antenna) << (2 * i)
    return value


def parse_beamform_payload(payload: bytes, *, apply_permutation: bool = True) -> ChannelSnapshot:
    """Decode one beamforming-report payload (the bytes after the code byte).

    Parameters
    ----------
    payload : bytes
        Header plus packed CSI.
    apply_permutation : bool, optional
        Reorder receive columns into physical antenna order.  Leave this on:
        the receive chain behind "column 1" changes from packet to packet, and
        without the reordering the antenna-pair axis of a feature matrix would
        mix antennas from row to row.

    Returns
    -------
    ChannelSnapshot
        The decoded report.

    Raises
    ------
    BeamformFormatError
        If the payload is too short or its CSI length disagrees with the
        number of chains.
    """
    if len(payload) < HEADER_SIZE:
        raise BeamformFormatError(
            f"payload of {len(payload)} bytes is shorter than the {HEADER_SIZE}-byte header"
        )
    fields = _HEADER.unpack_from(payload)
    timestamp_us, report_counter, n_rx, n_tx = fields[:4]
    rssi_db = fields[4:7]  # one value per receive chain
    noise_dbm, agc_db, antenna_selection, csi_length, rate_field = fields[7:]

    if not (1 <= n_tx <= 3 and 1 <= n_rx <= 3):
        raise BeamformFormatError(f"implausible chain counts n_tx={n_tx}, n_rx={n_rx}")
    expected = expected_csi_length(n_tx, n_rx)
    if csi_length != expected:
        raise BeamformFormatError(
            f"CSI length field is {csi_length} bytes but n_tx={n_tx}, n_rx={n_rx} "
            f"requires {expected}"
        )
    block = payload[HEADER_SIZE : HEADER_SIZE + csi_length]
    if len(block) < csi_length:
        raise BeamformFormatError(
            f"payload carries {len(block)} CSI bytes, header announces {csi_length}"
        )

    csi = unpack_csi(block, n_tx, n_rx)
    permutation, valid = decode_antenna_permutation(antenna_selection, n_rx)
    if apply_permutation and valid and n_rx > 1:
        # Reported column i came from physical antenna permutation[i].
        csi = csi[:, np.argsort(permutation), :]

    meta = RadioMeta(
        report_counter=report_counter,
        n_tx=n_tx,
        n_rx=n_rx,
        rssi_db=rssi_db,
        noise_dbm=noise_dbm,
        agc_db=agc_db,
        antenna_permutation=permutation,
        permutation_valid=valid,
        rate=decode_rate_flags(rate_field),
    )
    return ChannelSnapshot(csi=csi, timestamp_us=timestamp_us, meta=meta)


def encode_beamform_payload(snapshot: ChannelSnapshot) -> bytes:
    """Encode a snapshot into a beamforming-report payload.

    The snapshot's CSI is taken to be in physical antenna order (as produced
    by :func:`parse_beamform_payload` with ``apply_permutation=True``) and is
    reordered into reporting order according to its antenna permutation.

    Parameters
    ----------
    snapshot : ChannelSnapshot
        A report whose CSI components are integers in ``[-128, 127]``.

    Returns
    -------
    bytes
        Header plus packed CSI.
    """
    meta = snapshot.meta
    csi = snapshot.csi
    if meta.permutation_valid and meta.n_rx > 1:
        csi = csi[:, list(meta.antenna_permutation), :]
    block = pack_csi(csi)
    header = _HEADER.pack(
        snapshot.timestamp_us,
        meta.report_counter & 0xFFFF,
        meta.n_rx,
        meta.n_tx,
        *meta.rssi_db,
        meta.noise_dbm,
        meta.agc_db,
        encode_antenna_selection(meta.antenna_permutation),
        len(block),
        meta.rate.raw,
    )
    return header + block


def _frame(code: int, payload: bytes) -> bytes:
    """Wrap a payload in the length-prefixed frame format."""
    if len(payload) + 1 > 0xFFFF:
        raise BeamformFormatError(f"payload of {len(payload)} bytes does not fit in one frame")
    return _FRAME_PREFIX.pack(len(payload) + 1, code) + payload


class BeamformReportReader:
    """Stream beamforming reports out of a capture file.

    Reports are decoded one frame at a time, so a long capture is never held
    in memory as a whole.  Malformed reports are logged and skipped instead of
    aborting the read; :attr:`statistics` counts every kind of problem.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file.
    apply_permutation : bool, optional
        Reorder receive columns into physical antenna order (default True).

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from wifi_presence_sensing.synthetic import simple_snapshot
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = pathlib.Path(folder) / "capture.dat"
    ...     _ = write_beamform_file(path, [simple_snapshot(seed=s) for s in range(3)])
    ...     reader = BeamformReportReader(path)
    ...     n = sum(1 for _ in reader.iter_reports())
    >>> n, reader.statistics.n_reports
    (3, 3)
    """

    def __init__(self, path: str | Path, *, apply_permutation: bool = True) -> None:
        self.path = Path(path)
        self.apply_permutation = apply_permutation
        self.statistics = ReadStatistics()
        if not self.path.is_file():
            raise FileNotFoundError(f"capture file not found: {self.path}")

    def iter_reports(self) -> Iterator[ChannelSnapshot]:
        """Yield every decodable beamforming report in file order.

        Statistics are reset at the start of each iteration.
        """
        self.statistics = ReadStatistics()
        with self.path.open("rb") as stream:
            yield from self._iter_stream(stream)
        stats = self.statistics
        if stats.n_truncated or stats.n_malformed:
            _LOGGER.warning(
                "%s: skipped %d truncated and %d malformed report(s) out of %d frame(s)",
                self.path.name,
                stats.n_truncated,
                stats.n_malformed,
                stats.n_frames,
            )

    def _iter_stream(self, stream: BinaryIO) -> Iterator[ChannelSnapshot]:
        stats = self.statistics
        warned_permutation = False
        while True:
            prefix = stream.read(_FRAME_PREFIX.size)
            if not prefix:
                return
            if len(prefix) < _FRAME_PREFIX.size:
                stats.n_truncated += 1
                return
            frame_length, code = _FRAME_PREFIX.unpack(prefix)
            stats.n_frames += 1
            if frame_length < 1:
                _LOGGER.warning(
                    "%s: zero-length frame header at byte %d; stopping",
                    self.path.name,
                    stream.tell() - _FRAME_PREFIX.size,
                )
                stats.n_malformed += 1
                return
            payload = stream.read(frame_length - 1)
            if len(payload) < frame_length - 1:
                stats.n_truncated += 1
                return
            if code != BEAMFORM_REPORT_CODE:
                stats.n_other_codes += 1
                continue
            try:
                snapshot = parse_beamform_payload(payload, apply_permutation=self.apply_permutation)
            except BeamformFormatError as exc:
                stats.n_malformed += 1
                _LOGGER.warning("%s: frame %d skipped: %s", self.path.name, stats.n_frames, exc)
                continue
            if not snapshot.meta.permutation_valid:
                stats.n_invalid_permutation += 1
                if not warned_permutation:
                    warned_permutation = True
                    _LOGGER.warning(
                        "%s: antenna selection %s is not a permutation; such reports are "
                        "kept in reported order",
                        self.path.name,
                        snapshot.meta.antenna_permutation,
                    )
            stats.n_reports += 1
            yield snapshot

    def read_all(self) -> list[ChannelSnapshot]:
        """Decode the whole file into a list."""
        return list(self.iter_reports())


def read_beamform_file(
    path: str | Path, *, apply_permutation: bool = True
) -> tuple[list[ChannelSnapshot], ReadStatistics]:
    """Read every beamforming report of a capture file.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file.
    apply_permutation : bool, optional
        Reorder receive columns into physical antenna order (default True).

    Returns
    -------
    snapshots : list of ChannelSnapshot
        Decoded reports in file order.
    statistics : ReadStatistics
        What was read and what was skipped.
    """
    reader = BeamformReportReader(path, apply_permutation=apply_permutation)
    snapshots = reader.read_all()
    _LOGGER.info(
        "%s: %d report(s) decoded (%s)",
        reader.path.name,
        len(snapshots),
        ", ".join(f"{k}={v}" for k, v in reader.statistics.as_dict().items() if v),
    )
    return snapshots, reader.statistics


def write_beamform_file(
    path: str | Path,
    snapshots: Iterable[ChannelSnapshot],
    *,
    extra_frames: Iterable[tuple[int, bytes]] = (),
) -> int:
    """Write snapshots as a capture file readable by :func:`read_beamform_file`.

    Used to build synthetic fixtures and to export filtered captures.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination; parent directories are created.
    snapshots : iterable of ChannelSnapshot
        Reports with integer-valued CSI.
    extra_frames : iterable of (int, bytes), optional
        Additional ``(code, payload)`` frames appended after the reports,
        e.g. non-beamforming frames for testing.

    Returns
    -------
    int
        Number of bytes written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_frame(BEAMFORM_REPORT_CODE, encode_beamform_payload(s)) for s in snapshots]
    chunks.extend(_frame(code, payload) for code, payload in extra_frames)
    data = b"".join(chunks)
    destination.write_bytes(data)
    return len(data)
