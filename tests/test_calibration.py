"""Absolute scaling, the noise model and uniform resampling."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from wifi_presence_sensing.calibrate.noise_model import NoiseModel, NoiseStrategy, hampel_outliers
from wifi_presence_sensing.calibrate.resample import resample_uniform
from wifi_presence_sensing.calibrate.scaling import (
    amplitude_db,
    chain_power_correction,
    detrended_phase,
    from_db,
    to_calibrated_csi,
    to_db,
    total_received_power_dbm,
)


def test_db_conversions_are_inverse(rng):
    power = 10 ** rng.uniform(-9, 3, 50)
    np.testing.assert_allclose(from_db(to_db(power)), power)
    assert amplitude_db(np.array([0.0]))[0] == pytest.approx(-240.0)


def test_total_power_ignores_absent_chains_and_subtracts_agc():
    single = total_received_power_dbm(np.array([30.0, 0.0, 0.0]), 20.0)
    assert float(single) == pytest.approx(30 - 44 - 20)
    triple = total_received_power_dbm(np.array([[30.0, 30.0, 30.0]]), np.array([20.0]))
    assert float(triple[0]) == pytest.approx(30 + 10 * np.log10(3) - 64)
    assert np.isnan(total_received_power_dbm(np.zeros(3), 10.0))


def test_absolute_calibration_removes_agc_steps(rng):
    """A raw amplitude step caused by the AGC disappears after calibration."""
    channel = rng.normal(size=(2, 3, 30)) + 1j * rng.normal(size=(2, 3, 30))
    chain_power_dbm = -50.0
    reports, rssi, agc = [], [], []
    for gain_db in (30, 36, 24, 33):
        reports.append(channel * 10 ** (gain_db / 20))
        level = chain_power_dbm + 44 + gain_db
        rssi.append([level, level, level])
        agc.append(gain_db)
    calibrated = to_calibrated_csi(np.stack(reports), np.array(rssi), np.array(agc))
    raw_spread = np.ptp(amplitude_db(np.stack(reports)), axis=0).max()
    calibrated_spread = np.ptp(amplitude_db(calibrated), axis=0).max()
    assert raw_spread > 10.0
    assert calibrated_spread < 1e-9
    per_subcarrier_mw = (np.abs(calibrated[0]) ** 2).sum() / 30
    assert 10 * np.log10(per_subcarrier_mw) == pytest.approx(chain_power_dbm + 10 * np.log10(3))


def test_noise_reference_gives_snr_units():
    csi = np.full((1, 1, 30), 10.0 + 0j)
    rssi = np.array([40.0, 0.0, 0.0])
    agc = 30.0  # received power -34 dBm
    snr_units = to_calibrated_csi(
        csi, rssi, agc, noise_dbm=-94.0, reference="noise", include_quantization_noise=False
    )
    assert 10 * np.log10(np.mean(np.abs(snr_units) ** 2)) == pytest.approx(60.0)
    with_quantization = to_calibrated_csi(csi, rssi, agc, noise_dbm=-94.0, reference="noise")
    assert np.mean(np.abs(with_quantization) ** 2) < np.mean(np.abs(snr_units) ** 2)


def test_chain_power_correction_and_reference_validation():
    assert chain_power_correction(1) == 1.0
    assert chain_power_correction(2) == pytest.approx(np.sqrt(2.0))
    assert chain_power_correction(3) == pytest.approx(10 ** (4.5 / 20))
    with pytest.raises(ValueError):
        to_calibrated_csi(np.ones((1, 3, 30)), np.ones(3), 0.0, reference="noise")
    with pytest.raises(ValueError):
        to_calibrated_csi(np.ones((1, 3, 30)), np.ones(3), 0.0, reference="dbm")


def test_detrended_phase_removes_offset_and_slope_but_keeps_structure():
    k = np.arange(-28, 29, 2)
    bump = 0.4 * np.exp(-((k / 6.0) ** 2))
    csi = np.exp(1j * (1.3 + 0.25 * k + bump))
    phase = detrended_phase(csi, k)
    centred_bump = bump - np.polyval(np.polyfit(k, bump, 1), k)
    np.testing.assert_allclose(phase, centred_bump, atol=1e-10)


def test_hampel_rolling_and_global():
    values = np.r_[np.full(20, -92.0), -60.0, np.full(20, -91.0)]
    global_flags, _ = hampel_outliers(values)
    rolling_flags, reference = hampel_outliers(values, window=7)
    assert global_flags[20] and rolling_flags[20]
    assert global_flags.sum() == 1 and rolling_flags.sum() == 1
    assert reference[20] in (-92.0, -91.0)
    with pytest.raises(ValueError):
        hampel_outliers(values, window=4)


def test_noise_strategies(caplog):
    reported = np.r_[np.full(70, -92.0), np.full(28, -62.0), [-127.0, -127.0]]
    raw = NoiseModel(strategy="reported").estimate(reported)
    assert raw.replaced_fraction == 0.0
    guard = NoiseModel(strategy="sentinel_guard").estimate(reported)
    assert guard.replaced.sum() == 2 and np.all(guard.noise_dbm[-2:] == -92.0)
    with caplog.at_level(logging.WARNING):
        robust = NoiseModel(strategy=NoiseStrategy.ROBUST.value).estimate(reported)
    assert robust.replaced_fraction == pytest.approx(0.30)
    assert np.all(robust.noise_dbm == -92.0)
    assert "replaced 30.0%" in caplog.text
    with pytest.raises(ValueError):
        NoiseModel(strategy="median")


def test_robust_noise_is_clamped_to_the_plausible_range():
    model = NoiseModel(plausible_range_dbm=(-100.0, -85.0))
    estimate = model.estimate(np.full(10, -80.0))
    assert np.all(estimate.noise_dbm == -85.0)
    assert estimate.replaced_fraction == 1.0


def test_linear_resampling_is_exact_for_a_linear_signal():
    times = np.sort(np.random.default_rng(0).uniform(0, 10, 200))
    values = np.column_stack([3 * times + 1, -times])
    (segment,) = resample_uniform(times, values, 5.0, max_gap_s=1.0, method="linear")
    expected = np.column_stack([3 * segment.times_s + 1, -segment.times_s])
    np.testing.assert_allclose(segment.values, expected, atol=1e-9)
    assert segment.method == "linear"


def test_bursty_input_is_bin_averaged():
    bursts = np.concatenate([start + np.arange(20) * 1e-3 for start in np.arange(0, 5, 0.1)])
    values = np.where(np.arange(bursts.size) % 2 == 0, 1.0, -1.0)  # alternates inside each burst
    (segment,) = resample_uniform(bursts, values, 10.0, max_gap_s=0.5)
    assert segment.method == "bin_mean"
    assert np.allclose(segment.values, 0.0)


def test_pauses_split_and_short_segments_are_dropped(caplog):
    times = np.r_[np.arange(0, 3, 0.05), np.arange(10, 10.3, 0.05), np.arange(20, 26, 0.05)]
    values = np.sin(times)[:, None, None] * np.ones((1, 2, 3))
    segments = resample_uniform(times, values, 10.0, max_gap_s=1.0, min_duration_s=1.0)
    assert len(segments) == 2
    assert segments[1].start_s == pytest.approx(20.0)
    assert segments[0].values.shape[1:] == (2, 3)
    with caplog.at_level(logging.WARNING):
        assert resample_uniform(times, values, 10.0, max_gap_s=1.0, min_duration_s=100.0) == []
    assert "no segment" in caplog.text


def test_resampling_validates_input():
    with pytest.raises(ValueError):
        resample_uniform(np.array([1.0, 0.0]), np.zeros(2), 1.0, max_gap_s=1.0)
    with pytest.raises(ValueError):
        resample_uniform(np.arange(3.0), np.zeros(4), 1.0, max_gap_s=1.0)
    with pytest.raises(ValueError):
        resample_uniform(np.arange(3.0), np.zeros(3), 1.0, max_gap_s=1.0, method="cubic")
