from __future__ import annotations

from pathlib import Path

import pytest

from wifi_presence_sensing.config import Config, ConfigError
from wifi_presence_sensing.decompose import (
    CauchyStopping,
    FixedIterationStopping,
    ThreeThresholdStopping,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid_and_documented_choices():
    config = Config()
    assert config.decomposition.method == "eemd"
    assert config.decomposition.max_sift_iterations == 1000
    assert isinstance(config.decomposition.stopping_criterion(), ThreeThresholdStopping)
    assert config.evaluation.group_by == "session"
    assert config.localization.cdf_thresholds_m == (1, 2, 3, 4, 5, 6, 7, 8)


def test_yaml_round_trip(tmp_path: Path):
    config = Config().override({"resample.rate_hz": 5.0, "occupancy.feature_groups": ["modal"]})
    path = config.save(tmp_path / "config.yaml")
    loaded = Config.from_yaml(path)
    assert loaded == config
    assert loaded.occupancy.feature_groups == ("modal",)


def test_shipped_configuration_files_load():
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        assert isinstance(Config.from_yaml(path), Config), path


def test_default_yaml_documents_the_built_in_defaults():
    """Running without --config must mean exactly what configs/default.yaml says."""
    assert Config.from_yaml(PROJECT_ROOT / "configs" / "default.yaml") == Config()


def test_stopping_criteria_are_built_from_the_config():
    assert isinstance(
        Config().override({"decomposition.stopping": "cauchy"}).decomposition.stopping_criterion(),
        CauchyStopping,
    )
    fixed = Config().override(
        {"decomposition.stopping": "fixed", "decomposition.fixed_iterations": 8}
    )
    criterion = fixed.decomposition.stopping_criterion()
    assert isinstance(criterion, FixedIterationStopping) and criterion.n_iterations == 8


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("decomposition.method", "vmd", "decomposition.method"),
        ("decomposition.n_ensemble", 0, "n_ensemble"),
        ("decomposition.theta1", 0.9, "stopping parameters"),
        ("capture.tx_policy", "all", "tx_policy"),
        ("capture.bandwidth_mhz", 80, "bandwidth"),
        ("calibration.spatial_mapping", "phases", "mapping_phases_rad"),
        ("calibration.noise_strategy", "mean", "noise_strategy"),
        ("resample.rate_hz", 0, "rate_hz"),
        ("features.block_s", 1.0, "block_s"),
        ("occupancy.model", "xgboost", "occupancy.model"),
        ("occupancy.feature_groups", ["phase"], "feature_groups"),
        ("evaluation.n_splits", 1, "n_splits"),
        ("evaluation.group_by", "day", "group_by"),
        ("localization.confidence", 1.5, "confidence"),
        ("output.figure_format", "gif", "figure_format"),
        ("log_level", "LOUD", "log_level"),
        ("decomposition.n_ensemble", 2.5, "n_ensemble must be an integer"),
        ("decomposition.max_modes", 2.5, "max_modes must be an integer"),
        ("decomposition.fixed_iterations", 0, "fixed_iterations"),
        ("decomposition.paired_noise", "yes", "paired_noise must be true or false"),
        ("capture.apply_antenna_permutation", "no", "apply_antenna_permutation"),
        ("calibration.noise_window", 4, "noise_window must be odd"),
        ("calibration.noise_window", 1, "noise_window must be at least 3"),
        ("features.n_jobs", 0, "n_jobs"),
        ("table.n_streams", 0, "n_streams"),
        ("table.n_subcarriers", 30.0, "n_subcarriers must be an integer"),
        ("occupancy.n_estimators", True, "n_estimators must be an integer"),
        ("evaluation.inner_splits", 2.0, "inner_splits must be an integer"),
        ("evaluation.nested_search", 1, "nested_search must be true or false"),
        ("localization.cdf_thresholds_m", [3, 1], "strictly increasing"),
        ("localization.n_bootstrap", 10.5, "n_bootstrap must be an integer"),
        ("features.window_s", True, "features.window_s must be a number"),
        ("resample.rate_hz", "abc", "resample.rate_hz must be a number"),
        ("resample.min_segment_s", "ten", "min_segment_s must be a number"),
        ("decomposition.noise_scale", "0.2", "noise_scale must be a number"),
        ("decomposition.theta1", False, "theta1 must be a number"),
        ("calibration.noise_fallback_dbm", "low", "noise_fallback_dbm must be a number"),
        ("calibration.noise_plausible_range_dbm", [-60.0], "increasing \\[low, high\\] pair"),
        ("table.valid_range_db", [80.0, -40.0], "table.valid_range_db"),
        ("table.sentinel_values", 65535, "sentinel_values must be a list of numbers"),
        ("localization.cdf_thresholds_m", 5, "cdf_thresholds_m must be a list of numbers"),
        ("localization.cdf_thresholds_m", [1, "2"], "cdf_thresholds_m must be a number"),
        ("localization.confidence", "high", "confidence must be a number"),
        ("calibration.mapping_phases_rad", "abc", "mapping_phases_rad must be a square table"),
        ("calibration.mapping_phases_rad", [[0.0, 0.0], [0.0]], "square table"),
        ("calibration.mapping_phases_rad", [[0.0, "pi"], [0.0, 0.0]], "must be a number"),
    ],
)
def test_invalid_values_name_the_setting(key: str, value, message: str):
    with pytest.raises(ConfigError, match=message):
        Config().override({key: value})


def test_valid_counts_and_flags_are_accepted():
    config = Config().override(
        {
            "calibration.noise_window": 5,
            "features.n_jobs": -1,
            "capture.apply_antenna_permutation": False,
            "evaluation.nested_search": True,
        }
    )
    assert config.calibration.noise_window == 5
    assert config.features.n_jobs == -1
    assert config.capture.apply_antenna_permutation is False


def test_first_chain_policy_cannot_be_combined_with_a_spatial_mapping():
    with pytest.raises(ConfigError, match="first_chain"):
        Config().override(
            {"capture.tx_policy": "first_chain", "calibration.spatial_mapping": "hadamard"}
        )
    allowed = Config().override({"capture.tx_policy": "first_chain"})
    assert allowed.calibration.spatial_mapping == "none"


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        Config().override({"decomposition.ensemble_size": 10})
    with pytest.raises(ConfigError, match="unknown"):
        Config.from_dict({"decompositon": {}})
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"resample": {"rate": 3}})
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("does/not/exist.yaml")
