from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from wifi_presence_sensing.calibrate.link_quality import (
    MODULATION_MODELS,
    bpsk_error_rate,
    bpsk_error_rate_inverse,
    effective_snr,
    mmse_stream_snr,
    packet_effective_snr_db,
    qam16_error_rate,
    qam16_error_rate_inverse,
    qam64_error_rate,
    qam64_error_rate_inverse,
    qpsk_error_rate,
    qpsk_error_rate_inverse,
)

SNR_GRID = np.logspace(-3, 3, 61)


def q_function(x: np.ndarray) -> np.ndarray:
    return 0.5 * special.erfc(x / np.sqrt(2.0))


def test_bpsk_matches_the_textbook_q_function_form():
    np.testing.assert_allclose(
        bpsk_error_rate(SNR_GRID[:40]), q_function(np.sqrt(2 * SNR_GRID[:40]))
    )


@pytest.mark.parametrize(
    ("forward", "expected"),
    [
        (qpsk_error_rate, lambda g: q_function(np.sqrt(g))),
        (qam16_error_rate, lambda g: 0.75 * q_function(np.sqrt(g / 5.0))),
        (qam64_error_rate, lambda g: 7.0 / 12.0 * q_function(np.sqrt(g / 21.0))),
    ],
)
def test_qam_formulas(forward, expected):
    grid = SNR_GRID[:45]
    np.testing.assert_allclose(forward(grid), expected(grid), rtol=1e-10)


@pytest.mark.parametrize(
    ("forward", "inverse", "upper"),
    [
        (bpsk_error_rate, bpsk_error_rate_inverse, 1e2),
        (qpsk_error_rate, qpsk_error_rate_inverse, 1e2),
        (qam16_error_rate, qam16_error_rate_inverse, 1e3),
        (qam64_error_rate, qam64_error_rate_inverse, 1e3),
    ],
)
def test_linear_domain_round_trip(forward, inverse, upper):
    grid = SNR_GRID[np.less_equal(SNR_GRID, upper)]
    np.testing.assert_allclose(inverse(forward(grid)), grid, rtol=1e-6)


@pytest.mark.parametrize("name", sorted(MODULATION_MODELS))
def test_log_domain_round_trip_spans_seven_decades(name: str):
    model = MODULATION_MODELS[name]
    grid = np.logspace(-3, 4, 141)  # 1e-3 .. 1e4, where the linear-domain rate underflows
    np.testing.assert_allclose(
        model.snr_from_log_error_rate(model.log_error_rate(grid)), grid, rtol=1e-9
    )


def test_error_rates_decrease_with_snr_and_start_at_c_over_two():
    for model in MODULATION_MODELS.values():
        rates = model.error_rate(SNR_GRID)
        assert np.all(np.diff(rates) <= 0)
        assert float(model.error_rate(0.0)) == pytest.approx(model.max_error_rate)


def test_inverse_saturates_instead_of_returning_infinity():
    assert np.isfinite(bpsk_error_rate_inverse(0.0))
    assert float(bpsk_error_rate_inverse(0.5)) == pytest.approx(0.0, abs=1e-12)
    assert float(MODULATION_MODELS["qpsk"].snr_from_error_rate(0.9)) == 0.0
    with pytest.raises(ValueError):
        MODULATION_MODELS["qpsk"].snr_from_error_rate(-0.1)
    with pytest.raises(ValueError):
        bpsk_error_rate(-1.0)


@pytest.mark.parametrize("name", sorted(MODULATION_MODELS))
def test_flat_channel_keeps_its_snr_even_when_the_rate_underflows(name: str):
    for snr in (0.5, 10.0, 1e3, 1e4):
        assert float(effective_snr(np.full((3, 30), snr), name)) == pytest.approx(snr, rel=1e-9)


def test_effective_snr_lies_between_the_worst_subcarrier_and_the_mean(rng):
    snr = 10 ** rng.uniform(0, 3, size=(2, 30))
    for name in MODULATION_MODELS:
        value = float(effective_snr(snr, name))
        assert snr.min() <= value <= snr.mean()


def test_effective_snr_axis_argument():
    snr = np.stack([np.full(30, 4.0), np.full(30, 100.0)])
    np.testing.assert_allclose(effective_snr(snr, "qpsk", axis=1), [4.0, 100.0], rtol=1e-9)
    with pytest.raises(ValueError, match="unknown modulation"):
        effective_snr(snr, "qam256")


def test_mmse_snr_for_orthogonal_streams_is_the_per_stream_gain():
    channel = np.zeros((2, 2, 30), dtype=np.complex128)
    channel[0, 0] = 3.0  # stream 0 reaches rx 0 only: SNR 9
    channel[1, 1] = 2.0  # stream 1 reaches rx 1 only: SNR 4
    snr = mmse_stream_snr(channel)
    assert snr.shape == (30, 2)
    np.testing.assert_allclose(snr[:, 0], 9.0)
    np.testing.assert_allclose(snr[:, 1], 4.0)


def test_mmse_snr_drops_when_streams_interfere():
    channel = np.ones((2, 3, 30), dtype=np.complex128)  # identical, fully correlated streams
    assert np.all(mmse_stream_snr(channel) < 1.0)


def test_packet_effective_snr_in_db(rng):
    channel = 5.0 * (rng.normal(size=(4, 2, 3, 30)) + 1j * rng.normal(size=(4, 2, 3, 30)))
    result = packet_effective_snr_db(channel)
    assert set(result) == set(MODULATION_MODELS)
    for values in result.values():
        assert values.shape == (4,)
        assert np.all(np.isfinite(values))
