from __future__ import annotations

import logging

import numpy as np
import pytest

from wifi_presence_sensing.capture.snapshot import TIMESTAMP_PERIOD_US, ChannelSnapshot, RadioMeta
from wifi_presence_sensing.capture.subcarriers import (
    reported_subcarrier_indices,
    require_single_bandwidth,
    subcarrier_frequencies_hz,
)
from wifi_presence_sensing.capture.timeline import (
    contiguous_segments,
    find_gaps,
    sampling_diagnostics,
    unwrap_timestamps,
)
from wifi_presence_sensing.capture.trace import stack_snapshots
from wifi_presence_sensing.synthetic import simple_snapshot


def test_a_single_wrap_is_removed():
    raw = np.array([TIMESTAMP_PERIOD_US - 500, TIMESTAMP_PERIOD_US - 100, 300, 700])
    unwrapped = unwrap_timestamps(raw)
    assert np.all(np.diff(unwrapped) == [400, 400, 400])


def test_multiple_wraps_accumulate():
    step = TIMESTAMP_PERIOD_US // 3
    absolute = np.arange(0, 10 * step, step, dtype=np.int64)
    unwrapped = unwrap_timestamps(absolute % TIMESTAMP_PERIOD_US)
    np.testing.assert_array_equal(unwrapped, absolute)


def test_small_backward_steps_are_not_mistaken_for_wraps(caplog):
    raw = np.array([1_000, 2_000, 1_900, 3_000])
    with caplog.at_level(logging.WARNING):
        unwrapped = unwrap_timestamps(raw)
    np.testing.assert_array_equal(unwrapped, raw)
    assert "out of order" in caplog.text


def test_a_late_packet_across_the_wrap_is_not_a_second_wrap(caplog):
    period = TIMESTAMP_PERIOD_US
    raw = np.array([period - 300, period - 200, 50, period - 100, 150, 250])
    with caplog.at_level(logging.WARNING):
        unwrapped = unwrap_timestamps(raw)
    assert (unwrapped - unwrapped[0]).tolist() == [0, 100, 350, 200, 450, 550]
    assert "1 backwards timestamp step" in caplog.text


def test_a_long_pause_without_a_wrap_stays_a_pause(caplog):
    # 40 minutes is longer than half a counter period (about 35.8 minutes).
    pause = 2_400_000_000
    raw = np.array([1_000_000, 1_050_000, 1_050_000 + pause, 1_100_000 + pause])
    with caplog.at_level(logging.WARNING):
        unwrapped = unwrap_timestamps(raw)
    assert (unwrapped - unwrapped[0]).tolist() == [0, 50_000, 50_000 + pause, 100_000 + pause]
    assert "out of order" not in caplog.text


def test_a_reordered_wrap_keeps_the_trace_duration(occupied_snapshots):
    # Shift the capture so it crosses the wrap, then deliver the first
    # post-wrap packet before the last pre-wrap one.
    offset = TIMESTAMP_PERIOD_US - occupied_snapshots[len(occupied_snapshots) // 2].timestamp_us
    snapshots = [
        ChannelSnapshot(
            csi=s.csi, timestamp_us=(s.timestamp_us + offset) % TIMESTAMP_PERIOD_US, meta=s.meta
        )
        for s in occupied_snapshots
    ]
    i = int(np.flatnonzero(np.diff([s.timestamp_us for s in snapshots]) < 0)[0])
    snapshots[i], snapshots[i + 1] = snapshots[i + 1], snapshots[i]
    trace, _ = stack_snapshots(snapshots)
    in_order, _ = stack_snapshots(occupied_snapshots)
    assert np.ptp(trace.timestamps_s) == pytest.approx(np.ptp(in_order.timestamps_s))
    d = sampling_diagnostics(np.sort(trace.timestamps_s), max_gap_s=1.0)
    assert d.max_interval_s < 1.0


def test_unwrap_validates_its_input():
    with pytest.raises(ValueError):
        unwrap_timestamps(np.array([-1, 5]))
    with pytest.raises(ValueError):
        unwrap_timestamps(np.zeros((2, 2)))


def test_gaps_split_the_timeline():
    times = np.r_[np.arange(0, 2, 0.1), np.arange(5, 6, 0.1), [9.0]]
    assert find_gaps(times, 1.0).tolist() == [19, 29]
    segments = contiguous_segments(times, 1.0, min_samples=2)
    assert [(s.start, s.stop) for s in segments] == [(0, 20), (20, 30)]


def test_diagnostics_of_bursty_sampling_with_pauses():
    bursts = [start + np.arange(0, 0.02, 0.001) for start in np.arange(0, 30, 1.5)]
    times = np.concatenate(bursts)
    d = sampling_diagnostics(times, max_gap_s=1.0)
    assert d.burst_fraction > 0.9
    assert d.n_gaps == 19
    assert d.median_rate_hz > 900
    assert d.mean_rate_hz < 20
    assert any("pause" in line for line in d.describe_limits())


def test_one_hertz_sampling_cannot_resolve_walking_or_even_breathing():
    d = sampling_diagnostics(np.arange(0, 120, 1.0), max_gap_s=2.0)
    assert d.nyquist_hz == pytest.approx(0.5)
    resolvable = d.resolvable_bands()
    assert not resolvable["walking Doppler"]
    assert not resolvable["breathing"]
    assert any("NOT resolvable" in line for line in d.describe_limits())


def test_subcarrier_maps_differ_between_bandwidths():
    narrow = reported_subcarrier_indices(20)
    wide = reported_subcarrier_indices(40)
    assert narrow.size == wide.size == 30
    assert np.all(np.diff(narrow) > 0) and np.all(np.diff(wide) > 0)
    assert not np.array_equal(narrow, wide)
    freqs = subcarrier_frequencies_hz(20, 2.437e9)
    assert freqs[-1] - freqs[0] == pytest.approx(56 * 312_500.0)
    with pytest.raises(ValueError, match="mix channel bandwidths"):
        require_single_bandwidth([20, 40])
    with pytest.raises(ValueError):
        reported_subcarrier_indices(80)


def _with_tx(snapshot: ChannelSnapshot, n_tx: int, timestamp: int) -> ChannelSnapshot:
    meta = RadioMeta(**{**snapshot.meta.__dict__, "n_tx": n_tx})
    return ChannelSnapshot(csi=snapshot.csi[:n_tx], timestamp_us=timestamp, meta=meta)


def test_stacking_drops_minority_chain_counts_under_the_dominant_policy():
    base = simple_snapshot(seed=0)
    snapshots = [_with_tx(base, 1 if i % 5 == 0 else 2, 1000 * i) for i in range(20)]
    trace, report = stack_snapshots(snapshots, tx_policy="dominant")
    assert trace.csi.shape == (16, 2, 3, 30)
    assert report.dropped_chain_mismatch == 4
    assert report.radio_setups == {"20MHz/1x3": 4, "20MHz/2x3": 16}


def test_first_chain_policy_keeps_every_packet():
    base = simple_snapshot(seed=0)
    snapshots = [_with_tx(base, 1 if i % 5 == 0 else 2, 1000 * i) for i in range(20)]
    trace, report = stack_snapshots(snapshots, tx_policy="first_chain")
    assert trace.csi.shape == (20, 1, 3, 30)
    assert report.dropped_chain_mismatch == 0
    np.testing.assert_array_equal(trace.csi[1, 0], base.csi[0])


def test_stacking_unwraps_time_and_rejects_bad_input(occupied_snapshots):
    trace, _ = stack_snapshots(occupied_snapshots)
    assert trace.timestamps_s[0] == 0.0
    assert np.all(np.diff(trace.timestamps_s) > 0)
    assert trace.timestamps_s[-1] < 13.0
    with pytest.raises(ValueError):
        stack_snapshots([])
    with pytest.raises(ValueError):
        stack_snapshots(occupied_snapshots, tx_policy="all")


def test_snapshot_validates_shape():
    base = simple_snapshot(seed=0)
    with pytest.raises(ValueError):
        ChannelSnapshot(csi=base.csi[0], timestamp_us=0, meta=base.meta)
    with pytest.raises(ValueError):
        ChannelSnapshot(csi=base.csi, timestamp_us=TIMESTAMP_PERIOD_US, meta=base.meta)
    assert not base.csi.flags.writeable
