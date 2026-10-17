"""EMD, EEMD and CEEMDAN on signals with known components."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_presence_sensing.decompose import (
    CauchyStopping,
    FixedIterationStopping,
    SiftState,
    ThreeThresholdStopping,
    count_zero_crossings,
    decompose,
    decompose_ceemdan,
    decompose_eemd,
    decompose_emd,
    default_max_modes,
    extract_mode,
    extrema_envelopes,
    find_extrema,
    stopping_from_name,
)
from wifi_presence_sensing.decompose.sifting import natural_cubic_spline


def best_match(imfs: np.ndarray, component: np.ndarray) -> tuple[int, float]:
    scores = [abs(np.corrcoef(mode, component)[0, 1]) if np.any(mode) else 0.0 for mode in imfs]
    index = int(np.argmax(scores))
    return index, float(scores[index])


# --- sifting -----------------------------------------------------------------


def test_extrema_of_a_sine_and_plateaus():
    t = np.linspace(0, 2 * np.pi * 3, 600)
    e = find_extrema(np.sin(t))
    assert e.maxima.size == 3 and e.minima.size == 3
    plateau = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0])
    e = find_extrema(plateau)
    assert e.maxima.tolist() == [2]
    assert e.minima.tolist() == [5]
    assert find_extrema(np.arange(10.0)).count == 0


def test_zero_crossings():
    assert count_zero_crossings(np.sin(np.linspace(0.1, 6 * np.pi - 0.1, 500))) == 5
    assert count_zero_crossings(np.zeros(5)) == 0


def test_natural_spline_matches_linear_and_cubic_data():
    knots = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(
        natural_cubic_spline(knots, 2 * knots + 1, np.array([0.5, 2.0])), [2.0, 5.0]
    )
    np.testing.assert_allclose(natural_cubic_spline(knots[:2], [0.0, 4.0], np.array([0.25])), [1.0])


def test_envelopes_bracket_the_signal_inside_the_record():
    t = np.linspace(0, 1, 400)
    x = (1 + 0.5 * t) * np.sin(2 * np.pi * 9 * t)
    envelopes = extrema_envelopes(x)
    assert envelopes is not None
    inner = slice(20, -20)
    assert np.all(envelopes.upper[inner] >= x[inner] - 1e-2)
    assert np.all(envelopes.lower[inner] <= x[inner] + 1e-2)
    # Mirrored end extrema keep the envelopes from blowing up at the edges.
    assert np.max(np.abs(envelopes.upper)) < 2.5
    assert extrema_envelopes(np.linspace(0, 1, 50)) is None


# --- stopping criteria ----------------------------------------------------------


def state(candidate, mean, amplitude, *, previous=None, iteration=0, n_extrema=10, n_zero=10):
    return SiftState(candidate, previous, mean, amplitude, iteration, n_extrema, n_zero)


def test_three_threshold_rules():
    h = np.zeros(100)
    criterion = ThreeThresholdStopping()
    assert criterion.should_stop(state(h, np.zeros(100), np.ones(100)))
    large_mean = np.full(100, 0.2)
    assert not criterion.should_stop(state(h, large_mean, np.ones(100)))
    spike = np.zeros(100)
    spike[50] = 0.9  # under alpha but above theta2
    assert not criterion.should_stop(state(h, spike, np.ones(100)))
    assert not criterion.should_stop(state(h, np.zeros(100), np.ones(100), n_extrema=14, n_zero=10))
    with pytest.raises(ValueError):
        ThreeThresholdStopping(theta1=0.5, theta2=0.1)


def test_cauchy_and_fixed_rules():
    h = np.ones(10)
    cauchy = CauchyStopping(tolerance=0.2)
    assert not cauchy.should_stop(state(h, h, h))
    assert cauchy.should_stop(state(h, h, h, previous=h * 1.1))
    assert not cauchy.should_stop(state(h, h, h, previous=h * 3))
    fixed = FixedIterationStopping(4)
    assert not fixed.should_stop(state(h, h, h, iteration=3))
    assert fixed.should_stop(state(h, h, h, iteration=4))
    assert isinstance(stopping_from_name("cauchy", tolerance=0.1), CauchyStopping)
    with pytest.raises(ValueError):
        stopping_from_name("sd")


def test_sifting_cap_prevents_endless_loops():
    class Never:
        def should_stop(self, state):
            return state is None

    t = np.linspace(0, 1, 256)
    mode, steps = extract_mode(
        np.sin(2 * np.pi * 10 * t) + t, stopping=Never(), max_sift_iterations=7
    )
    assert steps == 7
    assert mode.shape == t.shape


# --- EMD -----------------------------------------------------------------------


def test_emd_separates_two_tones_and_reconstructs_exactly(two_tone):
    _t, fast, slow, trend = two_tone
    signal = fast + slow + trend
    result = decompose_emd(signal)
    np.testing.assert_allclose(result.reconstruct(), signal, atol=1e-10)
    fast_index, fast_score = best_match(result.imfs, fast)
    slow_index, slow_score = best_match(result.imfs, slow)
    assert fast_index == 0 and fast_score > 0.99
    assert slow_index > fast_index and slow_score > 0.9
    # The residual carries the trend.
    assert np.corrcoef(result.residual, trend)[0, 1] > 0.9


def test_emd_imfs_satisfy_the_extrema_zero_crossing_condition(two_tone):
    _, fast, slow, trend = two_tone
    result = decompose_emd(fast + slow + trend, stopping=ThreeThresholdStopping())
    for mode in result.imfs[:2]:
        assert abs(find_extrema(mode).count - count_zero_crossings(mode)) <= 2


def test_emd_edge_cases():
    assert decompose_emd(np.full(32, 3.0)).n_modes == 0
    monotone = decompose_emd(np.linspace(0, 1, 64) ** 2)
    assert monotone.n_modes == 0
    np.testing.assert_allclose(monotone.residual, np.linspace(0, 1, 64) ** 2)
    capped = decompose_emd(np.random.default_rng(0).normal(size=256), max_modes=2)
    assert capped.n_modes == 2
    with pytest.raises(ValueError):
        decompose_emd(np.zeros(3))
    with pytest.raises(ValueError):
        decompose_emd(np.array([0.0, 1.0, np.nan, 2.0, 3.0]))
    with pytest.raises(ValueError):
        decompose_emd(np.zeros((4, 4)))
    assert default_max_modes(1024) == 9


def test_emd_accepts_explicit_time(two_tone):
    t, fast, slow, _ = two_tone
    result = decompose_emd(fast + slow, time=t)
    assert best_match(result.imfs, fast)[1] > 0.99


@pytest.mark.parametrize("criterion", [CauchyStopping(), FixedIterationStopping(10)])
def test_emd_with_alternative_criteria(two_tone, criterion):
    _, fast, slow, _ = two_tone
    result = decompose_emd(fast + slow, stopping=criterion)
    assert best_match(result.imfs, fast)[1] > 0.98


# --- EEMD / CEEMDAN --------------------------------------------------------------


def test_eemd_is_reproducible_and_independent_of_n_jobs(two_tone):
    _, fast, slow, trend = two_tone
    x = (fast + slow + trend)[::2]
    a = decompose_eemd(x, n_ensemble=6, seed=11, max_modes=5)
    b = decompose_eemd(x, n_ensemble=6, seed=11, max_modes=5, n_jobs=2)
    c = decompose_eemd(x, n_ensemble=6, seed=12, max_modes=5)
    np.testing.assert_array_equal(a.imfs, b.imfs)
    assert not np.array_equal(a.imfs, c.imfs)
    np.testing.assert_allclose(a.reconstruct(), x, atol=1e-10)
    assert a.n_members == 6 and a.seed == 11


def test_eemd_separates_two_tones(two_tone):
    _, fast, slow, _ = two_tone
    x = fast + slow
    result = decompose_eemd(x, n_ensemble=20, noise_scale=0.1, seed=3)
    fast_index, fast_score = best_match(result.imfs, fast)
    slow_index, slow_score = best_match(result.imfs, slow)
    assert fast_score > 0.9 and slow_score > 0.8
    assert slow_index > fast_index


def test_residual_noise_shrinks_with_the_ensemble_size():
    x = np.sin(np.linspace(0, 12 * np.pi, 128))
    sizes = (4, 64)
    spread = [np.std(decompose_eemd(x, n_ensemble=n, seed=0).residual_noise) for n in sizes]
    expected_ratio = np.sqrt(sizes[1] / sizes[0])  # 1/sqrt(N) law
    assert spread[0] / spread[1] == pytest.approx(expected_ratio, rel=0.35)


def test_paired_noise_cancels_exactly():
    x = np.sin(np.linspace(0, 12 * np.pi, 128))
    result = decompose_eemd(x, n_ensemble=4, seed=0, paired_noise=True)
    assert result.n_members == 8
    assert np.max(np.abs(result.residual_noise)) < 1e-12


def test_eemd_validation_and_unseeded_warning(caplog):
    x = np.sin(np.linspace(0, 6, 64))
    with pytest.raises(ValueError):
        decompose_eemd(x, n_ensemble=0)
    with pytest.raises(ValueError):
        decompose_eemd(x, noise_scale=-1)
    decompose_eemd(x, n_ensemble=2)
    assert "not reproducible" in caplog.text


def test_ceemdan_is_complete_and_separates_tones(two_tone):
    _, fast, slow, trend = two_tone
    x = (fast + slow + trend)[::2]
    result = decompose_ceemdan(x, n_ensemble=10, seed=5, max_modes=6)
    np.testing.assert_allclose(result.reconstruct(), x, atol=1e-10)
    assert result.method == "ceemdan"
    assert best_match(result.imfs, fast[::2])[1] > 0.9
    again = decompose_ceemdan(x, n_ensemble=10, seed=5, max_modes=6)
    np.testing.assert_array_equal(result.imfs, again.imfs)


def test_dispatcher():
    x = np.sin(np.linspace(0, 20, 128)) + np.linspace(0, 1, 128)
    for method in ("emd", "eemd", "ceemdan"):
        result = decompose(x, method, n_ensemble=3, seed=1, max_modes=3)
        np.testing.assert_allclose(result.reconstruct(), x, atol=1e-10)
    with pytest.raises(ValueError):
        decompose(x, "vmd")
