"""End-to-end runs on simulated captures written to a temporary folder."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wifi_presence_sensing.calibrate import total_received_power_dbm
from wifi_presence_sensing.capture import DatasetManifest, ManifestError, write_beamform_file
from wifi_presence_sensing.config import Config
from wifi_presence_sensing.features import FeatureTable
from wifi_presence_sensing.pipeline import (
    build_dataset,
    build_table_dataset,
    calibrate_trace,
    evaluate_localization,
    evaluate_occupancy,
    evaluated_windows,
    inspect_capture,
    load_trace,
    occupancy_summary,
    spatial_mapping_for,
)
from wifi_presence_sensing.report import build_manifest, file_sha256, write_manifest
from wifi_presence_sensing.synthetic import (
    simple_snapshot,
    synthetic_amplitude_table,
    write_synthetic_dataset,
)


def test_inspect_reports_wrap_free_timing_and_noise_cleaning(
    tmp_path: Path, occupied_snapshots, fast_config: Config
):
    path = tmp_path / "capture.dat"
    write_beamform_file(path, occupied_snapshots)
    report, trace = inspect_capture(path, fast_config)
    assert report["read"]["n_reports"] == len(occupied_snapshots)
    # The capture starts 3 s before the counter wraps; the duration must stay positive.
    assert 11.0 < report["sampling"]["duration_s"] < 12.5
    assert report["sampling"]["mean_rate_hz"] > 15
    assert trace.timestamps_s[0] == 0.0
    assert set(report["median_effective_snr_db"]) == {"bpsk", "qpsk", "qam16", "qam64"}
    json.dumps(report)  # JSON-ready


def test_calibration_removes_agc_steps_from_a_simulated_trace(
    tmp_path: Path, geometry, fast_config: Config
):
    from wifi_presence_sensing.synthetic import CaptureScenario, make_environment, simulate_capture

    environment = make_environment(geometry, seed=1)
    scenario = CaptureScenario(duration_s=20.0, agc_step_probability=0.2, single_chain_fraction=0.0)
    path = tmp_path / "agc.dat"
    write_beamform_file(path, simulate_capture(scenario, geometry, environment, seed=2))
    trace, _, _ = load_trace(path, fast_config)
    calibrated = calibrate_trace(trace, fast_config)
    raw_db = 20 * np.log10(np.abs(trace.csi).mean(axis=(1, 2, 3)))
    calibrated_db = calibrated.amplitude_db.mean(axis=(1, 2, 3))
    assert np.ptp(trace.agc_db) >= 4
    assert np.std(calibrated_db) < 0.5 * np.std(raw_db)


def test_total_power_column_shares_the_calibration_reference(tmp_path: Path, fast_config: Config):
    """``total_dbm`` must agree with the power the CSI was scaled to, chain by chain."""
    snapshots = []
    for index in range(12):
        base = simple_snapshot(seed=index, timestamp_us=100_000 * (index + 1))
        # Chain c reports only in two packets out of three.
        rssi = (30 + index % 4, 32, 0 if index % 3 == 0 else 28)
        snapshots.append(replace(base, meta=replace(base.meta, rssi_db=rssi)))
    path = tmp_path / "intermittent_chain.dat"
    write_beamform_file(path, snapshots)
    trace, _, _ = load_trace(path, fast_config)
    calibrated = calibrate_trace(trace, fast_config)

    assert calibrated.power_names == ("chain_a_dbm", "chain_b_dbm", "total_dbm")
    expected = total_received_power_dbm(trace.rssi_db, trace.agc_db)
    np.testing.assert_allclose(calibrated.power_dbm[:, -1], expected)
    # The absolutely scaled CSI carries exactly that power per subcarrier.
    per_subcarrier_mw = (10.0 ** (calibrated.amplitude_db / 10.0)).sum(axis=(1, 2, 3)) / 30
    np.testing.assert_allclose(10.0 * np.log10(per_subcarrier_mw), expected, atol=1e-9)


def test_evaluated_windows_drop_only_block_straddlers(fast_config: Config, rng, caplog):
    n = 6
    table = FeatureTable(
        features=rng.normal(size=(n, 1)),
        feature_names=["modal__x"],
        labels=np.array(["a"] * n),
        recordings=np.array(["s/r"] * n),
        sessions=np.array(["s"] * n),
        blocks=np.array(["s/r#b0"] * n),
        window_start_s=np.array([0.0, 1.6, 3.2, 4.8, 6.4, 8.0]),
        positions=np.full((n, 2), np.nan),
    )
    config = fast_config.override({"features.window_s": 3.2, "features.block_s": 6.4})
    assert evaluated_windows(table, "session", config) is table
    kept = evaluated_windows(table, "block", config)
    assert kept.window_start_s.tolist() == [0.0, 1.6, 3.2, 6.4, 8.0]
    assert kept.blocks.tolist() == ["s/r#b0", "s/r#b0", "s/r#b0", "s/r#b1", "s/r#b1"]

    # A table that records its 3.2 s windows is purged with that length even
    # when the evaluation configuration claims shorter windows.
    recorded = replace(table, window_s=3.2)
    shorter = config.override({"features.window_s": 1.6})
    with caplog.at_level(logging.WARNING):
        kept = evaluated_windows(recorded, "block", shorter)
    assert kept.window_start_s.tolist() == [0.0, 1.6, 3.2, 6.4, 8.0]
    assert "3.2 s windows" in caplog.text
    # Without a recorded length the configured 1.6 s applies, and no window straddles a
    # block boundary.
    assert evaluated_windows(table, "block", shorter).n_windows == n


def test_spatial_mapping_setting(fast_config: Config):
    assert spatial_mapping_for(fast_config, 2) is None
    hadamard = fast_config.override({"calibration.spatial_mapping": "hadamard"})
    assert spatial_mapping_for(hadamard, 2).shape == (2, 2)
    assert spatial_mapping_for(hadamard, 1) is None
    phases = fast_config.override(
        {
            "calibration.spatial_mapping": "phases",
            "calibration.mapping_phases_rad": [[0.0, 0.0], [0.0, 3.14159]],
        }
    )
    assert spatial_mapping_for(phases, 2).shape == (2, 2)
    with pytest.raises(ValueError):
        spatial_mapping_for(phases, 3)


def test_occupancy_end_to_end(synthetic_occupancy: Path, fast_config: Config):
    manifest = DatasetManifest.from_yaml(synthetic_occupancy)
    assert manifest.labels == ["empty", "occupied"]
    config = fast_config.override({"evaluation.group_by": "block"})
    table, report = build_dataset(manifest, config)
    assert table.n_windows > 8
    assert {r["recording"] for r in report["recordings"]} == {r.name for r in manifest.recordings}
    assert all(r["read"]["n_malformed"] == 0 for r in report["recordings"])
    groups = {name.split("__")[0] for name in table.feature_names}
    assert groups == {"power", "response", "profile", "modal"}
    reports = evaluate_occupancy(table, config)
    summary = occupancy_summary(reports)
    assert list(summary["candidate"]) == ["modal", "raw_response", "majority", "mean_amplitude"]
    # A majority vote is chance level within every fold; pooled over folds it can dip below 0.5.
    assert summary.set_index("candidate").loc["majority", "uar"] <= 0.55


def test_localization_end_to_end(tmp_path: Path, fast_config: Config):
    manifest_path = write_synthetic_dataset(
        tmp_path / "loc",
        kind="localization",
        n_sessions=2,
        duration_s=8.0,
        grid_shape=(2, 2),
        seed=3,
    )
    manifest = DatasetManifest.from_yaml(manifest_path)
    assert manifest.has_positions
    table, _ = build_dataset(manifest, fast_config)
    assert table.has_positions
    benchmark = evaluate_localization(table, fast_config)
    assert set(benchmark.results) == {"received_power", "raw_response", "modal_response"}
    assert benchmark.n_folds == 2
    for result in benchmark.results.values():
        assert result.n_samples == table.n_windows


def test_table_dataset(tmp_path: Path, fast_config: Config):
    path = tmp_path / "table.csv"
    np.savetxt(
        path, synthetic_amplitude_table(rows_per_block=40, n_sentinels=5, seed=2), delimiter=","
    )
    config = fast_config.override(
        {
            "table.nominal_rate_hz": 5.0,
            "features.window_s": 4.0,
            "features.hop_s": 4.0,
            "features.block_s": 8.0,
        }
    )
    table, report = build_table_dataset(config, path)
    assert report["n_runs"] == 9
    assert report["n_sentinel"] == 5
    assert set(table.labels.tolist()) == {"2", "3", "4"}
    assert not any(name.startswith("power__") for name in table.feature_names)
    # A table carries no antenna identities, so its streams get neutral names.
    assert "profile__stream5_temporal_std_mean" in table.feature_names
    assert not any("tx0" in name for name in table.feature_names)
    assert np.all(np.isfinite(table.features))


def test_manifest_records_inputs_and_config(tmp_path: Path):
    data = tmp_path / "input.bin"
    data.write_bytes(b"abc")
    manifest = build_manifest(
        command="test",
        config=Config().to_dict(),
        inputs=[data, tmp_path / "missing.bin"],
        seed=3,
        extra={"array": np.arange(3), "nan": float("nan")},
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["inputs"][0]["sha256"] == file_sha256(data)
    assert loaded["inputs"][1]["missing"] is True
    assert loaded["details"]["array"] == [0, 1, 2]
    assert loaded["details"]["nan"] is None
    assert loaded["config"]["seed"] == 42
    assert loaded["created_utc"].endswith("+00:00")


def test_manifest_validation(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nrecordings:\n  - label: empty\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="path"):
        DatasetManifest.from_yaml(bad)
    duplicate = tmp_path / "dup.yaml"
    duplicate.write_text(
        "name: x\nrecordings:\n"
        "  - {path: a.dat, label: e, session: s}\n"
        "  - {path: a.dat, label: o, session: s}\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="twice"):
        DatasetManifest.from_yaml(duplicate)
    with pytest.raises(ManifestError, match="position"):
        DatasetManifest.from_dict(
            {"recordings": [{"path": "a.dat", "label": "e", "session": "s", "position": [1.0]}]},
            base=tmp_path,
        )


def test_manifest_requires_the_session_of_every_recording(tmp_path: Path):
    # Captures are laid out as session1/empty.dat, session2/empty.dat (as in
    # configs/datasets/example_occupancy.yaml and the synthesize output), so the
    # file stem cannot identify the session; each entry states it.
    entries = [
        {"path": f"session{k}/{label}.dat", "label": label}
        for k in (1, 2)
        for label in ("empty", "occupied")
    ]
    with pytest.raises(ManifestError, match="recording #0 is missing 'session'"):
        DatasetManifest.from_dict({"recordings": entries}, base=tmp_path)
    for k, entry in enumerate(entries):
        entry["session"] = f"session{k // 2 + 1}"
    manifest = DatasetManifest.from_dict({"recordings": entries}, base=tmp_path)
    assert [r.name for r in manifest.recordings] == [
        "session1/empty",
        "session1/occupied",
        "session2/empty",
        "session2/occupied",
    ]
