"""The command-line interface, driven end to end on simulated data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from wifi_presence_sensing.cli import app
from wifi_presence_sensing.features import FeatureTable

runner = CliRunner()

FAST = [
    "--set", "decomposition.n_ensemble=2",
    "--set", "decomposition.max_modes=2",
    "--set", "decomposition.stopping=fixed",
    "--set", "decomposition.fixed_iterations=5",
    "--set", "features.window_s=3.2",
    "--set", "features.hop_s=3.2",
    "--set", "features.block_s=6.4",
    "--set", "resample.min_segment_s=3.2",
    "--set", "evaluation.n_splits=2",
    "--set", "evaluation.group_by=block",
    "--set", "localization.n_bootstrap=20",
    "--set", "log_level=WARNING",
]  # fmt: skip


def invoke(*args: str):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cli")


def test_full_occupancy_workflow(workspace: Path):
    data = workspace / "data"
    out = invoke("synthesize", data, "--sessions", "2", "--duration", "14", "--seed", "1")
    assert "manifest" in out.output
    manifest = data / "manifest.yaml"
    assert manifest.is_file()

    inspected = invoke(
        "inspect", data / "session1" / "empty.dat", "--out", workspace / "inspect", *FAST
    )
    assert "packets" in inspected.output
    assert (workspace / "inspect" / "inspection.json").is_file()
    # Every figure comes with the table it plots.
    assert (workspace / "inspect" / "empty_intervals.png").is_file()
    intervals = pd.read_csv(workspace / "inspect" / "empty_intervals.csv")
    assert list(intervals.columns) == ["start_s", "interval_s"]
    assert (intervals["interval_s"] >= 0).all()

    invoke("prepare", "--dataset", manifest, "--out", workspace / "prepared", *FAST)
    features = workspace / "prepared" / "features.csv"
    assert features.is_file()
    manifest_json = json.loads(
        (workspace / "prepared" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_json["command"] == "prepare"
    assert all("sha256" in item for item in manifest_json["inputs"])

    evaluated = invoke("evaluate", features, "--out", workspace / "evaluate", *FAST)
    assert "modal" in evaluated.output
    assert (workspace / "evaluate" / "summary.csv").is_file()
    assert (workspace / "evaluate" / "confusion_modal.png").is_file()
    confusion = pd.read_csv(workspace / "evaluate" / "confusion_modal.csv")
    assert list(confusion.columns) == ["true_label", "predicted_empty", "predicted_occupied"]
    windows = len(pd.read_csv(workspace / "evaluate" / "out_of_fold_predictions.csv"))
    assert int(confusion[["predicted_empty", "predicted_occupied"]].to_numpy().sum()) == windows

    model = workspace / "model.joblib"
    invoke("train", features, "--model-out", model, *FAST)
    assert model.is_file()
    predictions = workspace / "predictions.csv"
    predicted = invoke("predict", model, features, "--out", predictions)
    assert "windows classified" in predicted.output
    assert predictions.read_text(encoding="utf-8").startswith("recording,window_start_s,predicted")

    decomposed = invoke(
        "decompose", data / "session1" / "occupied.dat", "--stream", "tx1rx2",
        "--out", workspace / "decompose", *FAST,
    )  # fmt: skip
    assert "IMF" in decomposed.output
    assert (workspace / "decompose" / "imfs.csv").is_file()


def test_inspect_keeps_captures_with_the_same_file_name_apart(workspace: Path):
    data = workspace / "same_names"
    invoke("synthesize", data, "--sessions", "2", "--duration", "6", "--seed", "2")
    captures = [data / "session1" / "occupied.dat", data / "session2" / "occupied.dat"]
    out = workspace / "inspect_same_names"
    result = invoke("inspect", *captures, "--out", out, *FAST)
    for capture in captures:
        assert f"{capture.as_posix()}: " in result.output
    for session in ("session1", "session2"):
        assert (out / f"{session}_occupied_intervals.png").is_file()
        assert (out / f"{session}_occupied_intervals.csv").is_file()
    reports = json.loads((out / "inspection.json").read_text(encoding="utf-8"))
    assert [report["path"] for report in reports] == [c.as_posix() for c in captures]


def test_train_and_predict_default_to_the_output_root(workspace: Path):
    features = workspace / "root_features.csv"
    unaligned_feature_table(features)
    root = workspace / "results"
    invoke("train", features, "--set", f"output.root={root.as_posix()}", *BLOCKS)
    model = root / "occupancy_model.joblib"
    assert model.is_file()
    invoke("predict", model, features)
    assert (root / "predictions.csv").is_file()


def test_localization_workflow(workspace: Path):
    data = workspace / "loc"
    invoke("synthesize", data, "--kind", "localization", "--sessions", "2", "--duration", "8")
    invoke("prepare", "--dataset", data / "manifest.yaml", "--out", workspace / "loc_prep", *FAST)
    result = invoke(
        "localize", workspace / "loc_prep" / "features.csv", "--out", workspace / "loc_out", *FAST
    )
    assert "modal_response" in result.output
    assert (workspace / "loc_out" / "error_cdf.csv").is_file()
    curve = pd.read_csv(workspace / "loc_out" / "error_curve.csv")
    assert set(curve["locator"]) >= {"modal_response", "raw_response"}
    assert (workspace / "loc_out" / "error_cdf.png").is_file()


def test_config_file_and_bad_overrides(workspace: Path):
    config_path = workspace / "custom.yaml"
    config_path.write_text(yaml.safe_dump({"resample": {"rate_hz": 0}}), encoding="utf-8")
    result = runner.invoke(app, ["prepare", "--config", str(config_path)])
    assert result.exit_code != 0
    result = runner.invoke(app, ["prepare", "--set", "decomposition.method=vmd"])
    assert result.exit_code != 0
    result = runner.invoke(app, ["prepare", "--set", "novalue"])
    assert result.exit_code != 0
    # Fractional counts, non-boolean flags and non-numeric lengths or rates are rejected
    # by the configuration, with the key in the message, instead of failing later in numpy.
    for override in (
        "decomposition.n_ensemble=2.5",
        "capture.apply_antenna_permutation=no",
        "features.window_s=true",
        "resample.rate_hz=abc",
    ):
        result = runner.invoke(app, ["prepare", "--set", override])
        assert result.exit_code == 2, result.output
        assert not isinstance(result.exception, TypeError)
        assert override.split("=")[0] in result.output
    result = runner.invoke(app, ["prepare", "--dataset", str(workspace / "nothing.yaml")])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "inspect",
        "synthesize",
        "prepare",
        "decompose",
        "train",
        "evaluate",
        "predict",
        "localize",
    ):
        assert command in result.output


def unaligned_feature_table(path: Path) -> FeatureTable:
    """Windows of 3.2 s starting at 1 s: every other one straddles a 6.4 s block boundary."""
    rng = np.random.default_rng(0)
    recordings, labels, sessions, starts, positions = [], [], [], [], []
    for session in range(4):
        for label, xy in (("empty", (1.0, 1.0)), ("occupied", (4.0, 2.0))):
            for k in range(6):
                recordings.append(f"session{session}/{label}")
                labels.append(label)
                sessions.append(f"session{session}")
                starts.append(1.0 + 3.2 * k)
                positions.append(xy)
    occupied = (np.array(labels) == "occupied").astype(float)
    n = len(labels)
    names = ["power__total_dbm_mean", "response__mean_db", "profile__p", "modal__m1"]
    table = FeatureTable(
        features=occupied[:, None] + rng.normal(0.0, 0.3, (n, len(names))),
        feature_names=names,
        labels=np.array(labels),
        recordings=np.array(recordings),
        sessions=np.array(sessions),
        blocks=np.array([f"{r}#b0" for r in recordings]),
        window_start_s=np.array(starts),
        positions=np.array(positions),
    )
    table.to_csv(path)
    return table


BLOCKS = [
    "--set", "features.window_s=3.2",
    "--set", "features.block_s=6.4",
    "--set", "evaluation.n_splits=2",
    "--set", "localization.n_bootstrap=20",
    "--set", "log_level=WARNING",
]  # fmt: skip


def test_block_grouping_reports_on_the_windows_it_kept(workspace: Path):
    """Windows dropped at block boundaries must not misalign the per-window outputs."""
    path = workspace / "unaligned.csv"
    table = unaligned_feature_table(path)
    kept = table.with_block_purge(window_s=3.2, block_s=6.4).n_windows
    assert 0 < kept < table.n_windows

    out = workspace / "evaluate_blocks"
    invoke("evaluate", path, "--set", "evaluation.group_by=block", "--out", out, *BLOCKS)
    predictions = pd.read_csv(out / "out_of_fold_predictions.csv")
    assert len(predictions) == kept
    details = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["details"]
    assert details["n_windows"] == kept
    assert details["n_windows_dropped_at_block_boundaries"] == table.n_windows - kept

    out = workspace / "localize_blocks"
    invoke("localize", path, "--set", "localization.group_by=block", "--out", out, *BLOCKS)
    estimates = pd.read_csv(out / "estimates.csv")
    assert len(estimates) == kept
    assert estimates[["true_x", "true_y"]].notna().all().all()
    # recording and window start identify a row of features.csv.
    kept_windows = table.with_block_purge(window_s=3.2, block_s=6.4)
    np.testing.assert_array_equal(estimates["recording"], kept_windows.recordings)
    np.testing.assert_allclose(estimates["window_start_s"], kept_windows.window_start_s)


def test_expected_failures_end_with_a_message_instead_of_a_traceback(workspace: Path):
    empty_capture = workspace / "no_reports.dat"
    empty_capture.write_bytes(b"")
    result = runner.invoke(app, ["inspect", str(empty_capture), "--out", str(workspace / "x")])
    assert result.exit_code == 2, result.output
    assert "no decodable" in result.output
    assert not isinstance(result.exception, ValueError)

    single_class = workspace / "single_class.csv"
    table = unaligned_feature_table(single_class)
    table.select(table.labels == "empty").to_csv(single_class)
    result = runner.invoke(
        app, ["train", str(single_class), "--model-out", str(workspace / "m.joblib")]
    )
    assert result.exit_code == 2, result.output
    assert "two classes" in result.output

    result = runner.invoke(app, ["predict", str(workspace / "absent.joblib"), str(single_class)])
    assert result.exit_code == 2, result.output
    assert "not found" in result.output


def test_module_entry_point_runs_the_cli():
    source = Path(__file__).resolve().parents[1] / "src"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(source), os.environ.get("PYTHONPATH")])),
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "wifi_presence_sensing", "--help"],
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "localize" in completed.stdout
