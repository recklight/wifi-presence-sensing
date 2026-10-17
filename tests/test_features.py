from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wifi_presence_sensing.config import DecompositionConfig, FeatureConfig
from wifi_presence_sensing.features import (
    MODAL_STATISTICS,
    PROFILE_METRICS,
    AmplitudeSeries,
    FeatureTable,
    build_feature_table,
    instantaneous_frequency,
    modal_feature_vector,
    mode_statistics,
    reduce_subcarriers,
    stream_names,
    subcarrier_profile,
    window_features,
    window_starts,
)


def test_instantaneous_frequency_of_a_chirp_rises():
    fs = 100.0
    t = np.arange(1000) / fs
    chirp = np.cos(2 * np.pi * (2.0 * t + 0.5 * t**2))  # 2 Hz -> 12 Hz
    frequency, amplitude = instantaneous_frequency(chirp, fs)
    assert np.median(frequency[50:150]) == pytest.approx(3.0, abs=0.3)
    assert np.median(frequency[850:950]) == pytest.approx(11.0, abs=0.3)
    assert np.median(amplitude) == pytest.approx(1.0, abs=0.05)


def test_stacked_modes_are_transformed_like_single_ones():
    fs = 40.0
    t = np.arange(400) / fs
    modes = np.stack(
        [np.sin(2 * np.pi * 2.5 * t), 0.3 * np.cos(2 * np.pi * 6.0 * t + 0.4), np.zeros_like(t)]
    )
    stacked_frequency, stacked_amplitude = instantaneous_frequency(modes, fs)
    assert stacked_frequency.shape == stacked_amplitude.shape == (3, 399)
    for row, mode in enumerate(modes):
        frequency, amplitude = instantaneous_frequency(mode, fs)
        np.testing.assert_allclose(stacked_frequency[row], frequency, atol=1e-12)
        np.testing.assert_allclose(stacked_amplitude[row], amplitude, atol=1e-12)
    # A padded all-zero mode has no weight and therefore a mean frequency of zero.
    stats = mode_statistics(modes, np.zeros_like(t), fs)
    assert stats["mean_frequency_hz"][:2] == pytest.approx([2.5, 6.0], abs=0.05)
    assert stats["mean_frequency_hz"][2] == 0.0


def test_mode_statistics_of_known_modes():
    fs = 50.0
    t = np.arange(500) / fs
    imfs = np.stack([np.sin(2 * np.pi * 5 * t), 0.5 * np.sin(2 * np.pi * 1 * t)])
    stats = mode_statistics(imfs, np.zeros_like(t), fs)
    assert stats["mean_frequency_hz"] == pytest.approx([5.0, 1.0], abs=0.1)
    assert stats["zero_crossing_rate_hz"] == pytest.approx([5.0, 1.0], abs=0.15)
    assert stats["energy_fraction"].sum() == pytest.approx(1.0)
    assert stats["energy_fraction"][0] == pytest.approx(0.8, abs=0.01)
    assert stats["variance"] == pytest.approx([0.5, 0.125], rel=0.02)


def test_modal_feature_vector_has_a_fixed_layout():
    fs = 10.0
    t = np.arange(100) / fs
    one_mode = np.sin(2 * np.pi * t)[None]
    values, names = modal_feature_vector(one_mode, 0.2 * t, fs, n_modes=3, prefix="modal__s")
    assert len(values) == len(names) == 3 * len(MODAL_STATISTICS) + 1
    assert names[-1] == "modal__s_trend_slope"
    assert values[-1] == pytest.approx(0.2)
    # The padded modes contribute zeros.
    assert np.all(values[len(MODAL_STATISTICS) : -1] == 0.0)
    with pytest.raises(ValueError):
        modal_feature_vector(one_mode, t, fs, n_modes=0)


def test_subcarrier_profile_metrics(rng):
    t = np.arange(200)
    coherent = np.sin(2 * np.pi * t / 25)[:, None] * np.ones(30) + 0.01 * rng.normal(size=(200, 30))
    independent = rng.normal(size=(200, 30))
    assert subcarrier_profile(coherent)["subcarrier_correlation"] > 0.95
    assert abs(subcarrier_profile(independent)["subcarrier_correlation"]) < 0.05
    shaped = np.ones((50, 30)) * np.linspace(0, 29, 30)
    metrics = subcarrier_profile(shaped)
    assert metrics["spectral_range_db"] == pytest.approx(29.0)
    assert metrics["temporal_std_max"] == 0.0
    assert set(metrics) == set(PROFILE_METRICS)
    with pytest.raises(ValueError):
        subcarrier_profile(np.ones((1, 30)))


def test_subcarrier_reduction():
    t = np.linspace(0, 1, 80)
    common = np.sin(2 * np.pi * 4 * t)
    window = 20.0 + np.outer(common, np.linspace(0.5, 1.5, 30))
    pca = reduce_subcarriers(window, "pca")
    mean = reduce_subcarriers(window, "mean")
    assert np.corrcoef(pca, common)[0, 1] > 0.999
    assert np.corrcoef(mean, common)[0, 1] > 0.999
    assert pca.mean() == pytest.approx(20.0)
    np.testing.assert_allclose(reduce_subcarriers(np.full((10, 30), 5.0)), 5.0)
    with pytest.raises(ValueError):
        reduce_subcarriers(window, "ica")


def test_window_starts():
    assert window_starts(100, 32, 16).tolist() == [0, 16, 32, 48, 64]
    assert window_starts(10, 32, 16).size == 0


def _series(
    label: str, recording: str, amplitude: np.ndarray, rate: float = 10.0
) -> AmplitudeSeries:
    return AmplitudeSeries(
        amplitude_db=amplitude,
        sampling_rate_hz=rate,
        stream_names=stream_names(1, 2),
        label=label,
        recording=recording,
        session=recording.split("/")[0],
        position=(1.0, 2.0),
    )


def test_window_features_names_and_groups(rng):
    decomposition = DecompositionConfig(method="emd", max_modes=2)
    window = rng.normal(size=(40, 2, 30))
    power = rng.normal(size=(40, 2))
    values, names = window_features(
        window, ["tx0rx0", "tx0rx1"], 10.0, decomposition, power_dbm=power, power_names=["a", "b"]
    )
    assert len(values) == len(names) == len(set(names))
    groups = {name.split("__")[0] for name in names}
    assert groups == {"power", "response", "profile", "modal"}
    assert "response__mean_db" in names
    assert sum(n.startswith("modal__tx0rx1_m") for n in names) == 2 * len(MODAL_STATISTICS)


def test_feature_table_is_deterministic_and_round_trips_csv(tmp_path: Path, rng):
    series = [
        _series("empty", "s1/a", 20 + 0.1 * rng.normal(size=(64, 2, 30))),
        _series("occupied", "s1/b", 20 + rng.normal(size=(64, 2, 30))),
    ]
    decomposition = DecompositionConfig(method="eemd", n_ensemble=2, max_modes=2, stopping="fixed")
    features = FeatureConfig(window_s=1.6, hop_s=1.6, block_s=3.2)
    table = build_feature_table(series, decomposition=decomposition, features=features, seed=4)
    again = build_feature_table(series, decomposition=decomposition, features=features, seed=4)
    np.testing.assert_array_equal(table.features, again.features)
    # 64 samples at 10 Hz -> 4 windows of 16; none straddles a 3.2 s block.
    assert table.n_windows == 8
    assert sorted(set(table.blocks.tolist())) == ["s1/a#b0", "s1/a#b1", "s1/b#b0", "s1/b#b1"]
    assert table.has_positions

    path = table.to_csv(tmp_path / "features.csv")
    loaded = FeatureTable.read_csv(path)
    np.testing.assert_allclose(loaded.features, table.features)
    assert loaded.feature_names == table.feature_names
    assert loaded.labels.tolist() == table.labels.tolist()
    assert loaded.groups("block").tolist() == table.groups("block").tolist()
    # The table remembers how long its windows are (16 samples at 10 Hz).
    assert table.window_s == loaded.window_s == pytest.approx(1.6)
    assert "window_s" not in loaded.feature_names
    assert set(loaded.feature_frame(["profile"]).columns) == {
        n for n in table.feature_names if n.startswith("profile__")
    }


def test_window_length_column_is_optional_but_must_be_consistent(tmp_path: Path, rng):
    series = [_series("x", "s1/a", 20 + rng.normal(size=(48, 2, 30)))]
    decomposition = DecompositionConfig(method="emd", max_modes=2)
    features = FeatureConfig(window_s=1.6, hop_s=1.6, block_s=3.2)
    table = build_feature_table(series, decomposition=decomposition, features=features, seed=0)
    frame = table.to_frame()
    assert FeatureTable.from_frame(frame.drop(columns=["window_s"])).window_s is None
    frame.loc[0, "window_s"] = 3.2
    with pytest.raises(ValueError, match="different lengths"):
        FeatureTable.from_frame(frame)


def test_windows_straddling_blocks_are_purged_before_block_grouping(rng):
    series = [_series("x", "s1/a", rng.normal(size=(100, 2, 30)))]
    decomposition = DecompositionConfig(method="emd", max_modes=2)
    features = FeatureConfig(window_s=2.0, hop_s=1.0, block_s=5.0)
    table = build_feature_table(series, decomposition=decomposition, features=features, seed=0)
    assert table.n_windows == 9  # starts 0, 1, ..., 8 s
    purged = table.with_block_purge(window_s=2.0, block_s=5.0)
    starts = purged.window_start_s
    assert starts.tolist() == [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0]
    assert np.all(np.floor(starts / 5.0) == np.floor((starts + 1.9) / 5.0))
    assert sorted(set(purged.blocks.tolist())) == ["s1/a#b0", "s1/a#b1"]
    with pytest.raises(ValueError):
        table.with_block_purge(window_s=2.0, block_s=1.0)


def test_feature_table_rejects_inconsistent_series(rng):
    decomposition = DecompositionConfig(method="emd", max_modes=2)
    features = FeatureConfig(window_s=1.6, hop_s=1.6, block_s=3.2)
    good = _series("a", "s1/a", rng.normal(size=(32, 2, 30)))
    other = AmplitudeSeries(
        amplitude_db=rng.normal(size=(32, 3, 30)),
        sampling_rate_hz=10.0,
        stream_names=stream_names(1, 3),
        label="b",
        recording="s1/c",
        session="s1",
    )
    with pytest.raises(ValueError, match="antenna pairs"):
        build_feature_table([good, other], decomposition=decomposition, features=features, seed=0)
    with pytest.raises(ValueError, match="no 10 s window"):
        build_feature_table(
            [good],
            decomposition=decomposition,
            features=FeatureConfig(window_s=10.0, hop_s=1.0, block_s=20.0),
            seed=0,
        )
    with pytest.raises(ValueError, match="at least 16"):
        build_feature_table(
            [good],
            decomposition=decomposition,
            features=FeatureConfig(window_s=1.0, hop_s=1.0, block_s=2.0),
            seed=0,
        )
