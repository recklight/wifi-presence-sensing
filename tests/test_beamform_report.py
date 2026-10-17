"""Bit-level decoding of beamforming reports, checked against an independent reference."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import pytest

from wifi_presence_sensing.capture.beamform_report import (
    BeamformFormatError,
    BeamformReportReader,
    decode_antenna_permutation,
    encode_beamform_payload,
    expected_csi_length,
    pack_csi,
    parse_beamform_payload,
    read_beamform_file,
    unpack_csi,
    write_beamform_file,
)
from wifi_presence_sensing.capture.rate_flags import decode_rate_flags, encode_rate_flags
from wifi_presence_sensing.capture.snapshot import ChannelSnapshot, RadioMeta
from wifi_presence_sensing.synthetic import simple_snapshot


def reference_unpack(block: bytes, n_tx: int, n_rx: int) -> np.ndarray:
    """Straightforward per-value decoder written from the format description.

    Kept deliberately naive (a Python loop over the bit stream) so the
    vectorized decoder is compared with something structurally different.
    """
    data = [*block, 0]
    out = np.zeros((n_tx, n_rx, 30), dtype=np.complex128)
    index = 0
    for group in range(30):
        index += 3
        for entry in range(n_tx * n_rx):
            parts = []
            for offset in (0, 8):
                position = index + offset
                byte, shift = divmod(position, 8)
                value = ((data[byte] >> shift) | (data[byte + 1] << (8 - shift))) & 0xFF
                parts.append(value - 256 if value >= 128 else value)
            tx, rx = entry % n_tx, entry // n_tx
            out[tx, rx, group] = parts[0] + 1j * parts[1]
            index += 16
    return out


@pytest.mark.parametrize(("n_tx", "n_rx"), [(1, 1), (1, 3), (2, 3), (3, 3), (2, 2)])
def test_vectorized_unpacking_matches_the_reference(n_tx: int, n_rx: int, rng):
    block = rng.integers(0, 256, expected_csi_length(n_tx, n_rx), dtype=np.uint8).tobytes()
    np.testing.assert_array_equal(
        unpack_csi(block, n_tx, n_rx), reference_unpack(block, n_tx, n_rx)
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(("n_tx", "n_rx"), [(1, 1), (1, 3), (2, 2), (2, 3), (3, 3)])
def test_pack_and_unpack_round_trip_including_the_extremes(n_tx: int, n_rx: int, seed: int):
    rng = np.random.default_rng(seed)
    real = rng.integers(-128, 128, (n_tx, n_rx, 30))
    imag = rng.integers(-128, 128, (n_tx, n_rx, 30))
    real.flat[0], imag.flat[-1] = -128, 127
    csi = real + 1j * imag
    np.testing.assert_array_equal(unpack_csi(pack_csi(csi), n_tx, n_rx), csi)


def test_handcrafted_first_entry_straddles_a_byte_boundary():
    # 3 padding bits, then real = -2 (0xFE) and imaginary = 5 (0x05), LSB first.
    bits = [0, 0, 0] + [(0xFE >> b) & 1 for b in range(8)] + [(0x05 >> b) & 1 for b in range(8)]
    bits += [0] * (expected_csi_length(1, 1) * 8 - len(bits))
    block = np.packbits(np.array(bits, dtype=np.uint8), bitorder="little").tobytes()
    assert unpack_csi(block, 1, 1)[0, 0, 0] == -2 + 5j


def test_pack_rejects_values_that_do_not_fit_in_eight_bits():
    with pytest.raises(BeamformFormatError):
        pack_csi(np.full((1, 1, 30), 200 + 0j))
    with pytest.raises(BeamformFormatError):
        pack_csi(np.full((1, 1, 30), 1.5 + 0j))


def test_payload_round_trip_restores_physical_antenna_order():
    original = simple_snapshot(seed=4)
    decoded = parse_beamform_payload(encode_beamform_payload(original))
    np.testing.assert_array_equal(decoded.csi, original.csi)
    assert decoded.meta == original.meta
    assert decoded.timestamp_us == original.timestamp_us


def test_permutation_is_applied_when_decoding():
    snapshot = simple_snapshot(seed=9)
    payload = encode_beamform_payload(snapshot)
    raw = parse_beamform_payload(payload, apply_permutation=False)
    permutation = snapshot.meta.antenna_permutation
    # Reported column i holds physical antenna permutation[i].
    for column, antenna in enumerate(permutation):
        np.testing.assert_array_equal(raw.csi[:, column, :], snapshot.csi[:, antenna, :])


def test_invalid_permutation_is_flagged_and_left_unapplied(tmp_path: Path, caplog):
    good = simple_snapshot(seed=2)
    meta = RadioMeta(
        **{**good.meta.__dict__, "antenna_permutation": (0, 0, 1), "permutation_valid": False}
    )
    bad = ChannelSnapshot(csi=good.csi, timestamp_us=5, meta=meta)
    path = tmp_path / "bad_perm.dat"
    write_beamform_file(path, [bad])
    with caplog.at_level(logging.WARNING):
        snapshots, stats = read_beamform_file(path)
    assert stats.n_invalid_permutation == 1
    assert snapshots[0].meta.permutation_valid is False
    np.testing.assert_array_equal(snapshots[0].csi, good.csi)
    assert decode_antenna_permutation(0b010000, 3)[1] is False


def test_single_transmit_chain_reports_keep_three_dimensions(tmp_path: Path):
    snapshot = simple_snapshot(seed=1, n_tx=1)
    path = tmp_path / "single.dat"
    write_beamform_file(path, [snapshot])
    (decoded,), _ = read_beamform_file(path)
    assert decoded.csi.shape == (1, 3, 30)


def test_reader_skips_other_codes_malformed_and_truncated_frames(tmp_path: Path, caplog):
    snapshots = [simple_snapshot(seed=s, timestamp_us=1000 * s) for s in range(4)]
    path = tmp_path / "mixed.dat"
    good = encode_beamform_payload(snapshots[0])
    malformed = bytearray(good)
    struct.pack_into("<H", malformed, 16, 999)  # wrong CSI length field
    write_beamform_file(
        path,
        snapshots,
        extra_frames=[(0xC1, b"\x00" * 12), (0xBB, bytes(malformed)), (0xBB, b"\x01\x02")],
    )
    data = path.read_bytes()
    # Append a frame header promising more bytes than the file holds.
    path.write_bytes(data + struct.pack(">HB", 400, 0xBB) + b"\x00" * 50)

    with caplog.at_level(logging.WARNING):
        reader = BeamformReportReader(path)
        decoded = reader.read_all()
    stats = reader.statistics
    assert len(decoded) == 4
    assert stats.n_reports == 4
    assert stats.n_other_codes == 1
    assert stats.n_malformed == 2
    assert stats.n_truncated == 1
    assert "skipped" in caplog.text


def test_reader_streams_reports_lazily(tmp_path: Path):
    path = tmp_path / "stream.dat"
    write_beamform_file(path, [simple_snapshot(seed=s) for s in range(5)])
    iterator = BeamformReportReader(path).iter_reports()
    first = next(iterator)
    assert first.csi.shape == (2, 3, 30)
    assert sum(1 for _ in iterator) == 4


def test_simulated_capture_survives_a_write_read_cycle(tmp_path: Path, occupied_snapshots):
    path = tmp_path / "capture.dat"
    write_beamform_file(path, occupied_snapshots)
    decoded, stats = read_beamform_file(path)
    assert stats.n_reports == len(occupied_snapshots)
    for original, restored in zip(occupied_snapshots, decoded, strict=True):
        np.testing.assert_array_equal(restored.csi, original.csi)
        assert restored.meta.rssi_db == original.meta.rssi_db


def test_missing_file_and_bad_header_raise_clear_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BeamformReportReader(tmp_path / "absent.dat")
    with pytest.raises(BeamformFormatError, match="shorter than"):
        parse_beamform_payload(b"\x00" * 10)


def test_metadata_fields_are_decoded():
    meta = RadioMeta(
        report_counter=77,
        n_tx=2,
        n_rx=3,
        rssi_db=(40, 0, 33),
        noise_dbm=-95,
        agc_db=44,
        antenna_permutation=(2, 0, 1),
        permutation_valid=True,
        rate=decode_rate_flags(encode_rate_flags(14, bandwidth_mhz=40)),
    )
    snapshot = ChannelSnapshot(csi=np.ones((2, 3, 30)), timestamp_us=2**32 - 1, meta=meta)
    decoded = parse_beamform_payload(encode_beamform_payload(snapshot))
    assert decoded.meta == meta
    assert decoded.meta.bandwidth_mhz == 40
    assert decoded.timestamp_us == 2**32 - 1
