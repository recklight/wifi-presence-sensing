"""Shared fixtures.  Every fixture is synthesized in code; no measurement file is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wifi_presence_sensing.config import Config
from wifi_presence_sensing.synthetic import (
    CaptureScenario,
    RoomGeometry,
    make_environment,
    simulate_capture,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def two_tone() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A fast tone, a slow tone and a linear trend, sampled at 128 Hz for 4 s."""
    t = np.arange(512) / 128.0
    fast = np.sin(2 * np.pi * 16.0 * t)
    slow = 0.8 * np.sin(2 * np.pi * 1.5 * t)
    trend = 0.3 * t
    return t, fast, slow, trend


@pytest.fixture
def geometry() -> RoomGeometry:
    return RoomGeometry()


@pytest.fixture
def occupied_snapshots(geometry: RoomGeometry):
    environment = make_environment(geometry, seed=3)
    scenario = CaptureScenario(
        duration_s=12.0,
        packet_rate_hz=20.0,
        person_xy=(4.0, 3.2),
        start_timestamp_us=2**32 - 3_000_000,
    )
    return simulate_capture(scenario, geometry, environment, seed=7)


@pytest.fixture
def fast_config() -> Config:
    """A configuration small enough for unit tests to run in seconds."""
    return Config().override(
        {
            "decomposition.method": "eemd",
            "decomposition.n_ensemble": 3,
            "decomposition.max_modes": 3,
            "decomposition.stopping": "fixed",
            "decomposition.fixed_iterations": 6,
            "features.window_s": 3.2,
            "features.hop_s": 3.2,
            "features.block_s": 6.4,
            "resample.rate_hz": 10.0,
            "resample.min_segment_s": 3.2,
            "evaluation.n_splits": 2,
            "localization.n_bootstrap": 50,
            "log_level": "WARNING",
        }
    )


@pytest.fixture
def synthetic_occupancy(tmp_path: Path) -> Path:
    from wifi_presence_sensing.synthetic import write_synthetic_dataset

    return write_synthetic_dataset(
        tmp_path / "occupancy", kind="occupancy", n_sessions=2, duration_s=14.0, seed=5
    )
