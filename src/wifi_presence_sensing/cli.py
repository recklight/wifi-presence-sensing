"""Command-line interface.

Commands
--------
``inspect``
    Decode capture files and report packets, radio setup, sampling limits,
    noise-floor quality and effective SNR.
``synthesize``
    Write simulated captures and a dataset manifest for a dry run.
``prepare``
    Build the windowed feature table from a dataset manifest or a CSV
    amplitude table.
``decompose``
    Decompose one antenna pair of one capture and write its IMFs.
``train``
    Fit the occupancy classifier on a feature table and save it.
``evaluate``
    Grouped cross-validation of the occupancy classifier against baselines
    (UAR, sensitivity, specificity, confusion matrix, ROC-AUC).
``predict``
    Apply a saved classifier to a feature table.
``localize``
    Grouped cross-validated comparison of the three fingerprint locators,
    with error-distance CDFs and confidence bands.

Options
-------
Every command except ``synthesize`` and ``predict`` reads the configuration
from ``--config`` and repeated ``--set key=value`` overrides.  ``inspect``,
``prepare``, ``decompose``, ``evaluate`` and ``localize`` write into an output
folder (``--out``, or a new time-stamped folder under ``output.root``) together
with ``config.yaml`` and a ``manifest.json``.  ``train`` stores the
configuration inside the model file (``--model-out``, by default under
``output.root``), ``predict`` writes one CSV file (``--out``, by default under
the ``output.root`` saved with the model) and ``synthesize`` writes captures
plus their dataset manifest.

Expected failures, such as a missing file or a feature table that cannot be
split as requested, end with a one-line error message and exit code 2; the
traceback is logged at DEBUG level.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from .config import Config, ConfigError, OutputConfig
from .logging_utils import configure_logging

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Local variables of a failing frame can hold whole captures; never dump them.
    pretty_exceptions_show_locals=False,
    help="WiFi CSI analysis: beamforming-report parsing, calibration, EEMD features, "
    "occupancy classification and fingerprint localization.",
)

ConfigOption = typer.Option(None, "--config", "-c", help="YAML configuration file.")
SetOption = typer.Option(
    None, "--set", "-s", help="Override one config value: key=value (repeatable)."
)


def _coerce(raw: str) -> Any:
    """Parse a ``--set`` value: true/false, none/null, then JSON, else the raw string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Report the library's expected failures as a usage error, not a traceback.

    Bad inputs surface deep inside the library as ``ValueError`` (including
    :class:`~wifi_presence_sensing.config.ConfigError` and the split errors) or
    ``FileNotFoundError``; both are caught here.
    """
    try:
        yield
    except (ValueError, FileNotFoundError) as exc:
        _LOGGER.debug("command failed", exc_info=True)
        raise typer.BadParameter(str(exc)) from None


def _load_config(config_path: Path | None, overrides: list[str] | None) -> Config:
    try:
        config = Config.from_yaml(config_path) if config_path is not None else Config()
        if overrides:
            parsed: dict[str, Any] = {}
            for item in overrides:
                if "=" not in item:
                    raise typer.BadParameter(
                        f"--set expects key=value, got {item!r}", param_hint="--set"
                    )
                key, _, raw = item.partition("=")
                parsed[key.strip()] = _coerce(raw.strip())
            config = config.override(parsed)
    except (ConfigError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from None
    configure_logging(config.log_level)
    return config


def _run_folder(config: Config, label: str, out: Path | None) -> Path:
    from .report.manifest import run_directory

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        return out
    return run_directory(config.output.root, label)


def _output_stems(captures: list[Path]) -> list[str]:
    """Names for the per-capture outputs of ``inspect``, distinct for every capture.

    Captures are usually laid out as ``session1/occupied.dat``,
    ``session2/occupied.dat``, so a repeated file stem gets its folder name in
    front; a stem that still repeats gets the capture's position as well.

    Examples
    --------
    >>> _output_stems([Path("s1/occupied.dat"), Path("s2/occupied.dat"), Path("s2/empty.dat")])
    ['s1_occupied', 's2_occupied', 'empty']
    """
    stems = [capture.stem for capture in captures]
    names = [
        f"{capture.parent.name}_{capture.stem}" if stems.count(capture.stem) > 1 else capture.stem
        for capture in captures
    ]
    return [
        f"{index}_{name}" if names.count(name) > 1 else name for index, name in enumerate(names)
    ]


def _finish(folder: Path, command: str, config: Config, inputs: list[Path], details: dict) -> None:
    from .report.manifest import build_manifest, write_manifest

    config.save(folder / "config.yaml")
    write_manifest(
        folder / "manifest.json",
        build_manifest(
            command=command,
            config=config.to_dict(),
            inputs=inputs,
            seed=config.seed,
            extra=details,
        ),
    )


@app.command()
def inspect(
    captures: list[Path] = typer.Argument(..., help="Capture files (.dat)."),
    config_file: Path = ConfigOption,
    set_: list[str] = SetOption,
    out: Path = typer.Option(
        None, "--out", "-o", help="Output folder (default: a new run folder)."
    ),
) -> None:
    """Report what each capture contains and what its sampling can resolve."""
    from .pipeline import inspect_capture
    from .report.figures import packet_interval_table, plot_sampling_intervals, save_figure
    from .report.manifest import write_json

    config = _load_config(config_file, set_)
    missing = [capture for capture in captures if not capture.is_file()]
    if missing:
        raise typer.BadParameter(f"capture not found: {missing[0]}")
    folder = _run_folder(config, "inspect", out)
    reports = []
    for capture, name in zip(captures, _output_stems(captures), strict=True):
        with _reported_errors():
            report, trace = inspect_capture(capture, config)
        reports.append(report)
        # With repeated file names, the path as given tells the captures apart.
        label = capture.name if name == capture.stem else capture.as_posix()
        stem = folder / f"{name}_intervals"
        save_figure(
            plot_sampling_intervals(trace.timestamps_s, title=f"{label}: packet intervals"),
            stem,
            config.output.figure_format,
        )
        packet_interval_table(trace.timestamps_s).to_csv(f"{stem}.csv", index=False)
        sampling = report["sampling"]
        shape = report["shape"]
        replaced = report["noise"]["replaced_fraction"]
        typer.echo(
            f"{label}: {shape['n_packets']} packets "
            f"({shape['n_tx']}x{shape['n_rx']}, {report['bandwidth_mhz']} MHz), "
            f"{sampling['duration_s']:.1f} s, mean rate {sampling['mean_rate_hz']:.3g} Hz, "
            f"{sampling['n_gaps']} gap(s), noise replaced {replaced:.1%}"
        )
        for line in report["sampling_limits"][1:]:
            typer.echo(f"    {line}")
    write_json(folder / "inspection.json", reports)
    _finish(folder, "inspect", config, list(captures), {"n_captures": len(captures)})
    typer.echo(f"written to {folder}")


@app.command()
def synthesize(
    output_dir: Path = typer.Argument(..., help="Folder for the simulated captures."),
    kind: str = typer.Option("occupancy", "--kind", help="occupancy or localization."),
    sessions: int = typer.Option(3, "--sessions", help="Number of recording sessions."),
    duration: float = typer.Option(40.0, "--duration", help="Seconds per recording."),
    rate: float = typer.Option(20.0, "--rate", help="Mean packet rate in Hz."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
) -> None:
    """Write simulated captures and their dataset manifest (no real data involved)."""
    from .synthetic import write_synthetic_dataset

    configure_logging("INFO")
    with _reported_errors():
        manifest = write_synthetic_dataset(
            output_dir,
            kind=kind,
            n_sessions=sessions,
            duration_s=duration,
            packet_rate_hz=rate,
            seed=seed,
        )
    typer.echo(f"manifest: {manifest}")


@app.command()
def prepare(
    dataset: Path = typer.Option(None, "--dataset", "-d", help="Dataset manifest (YAML)."),
    table: Path = typer.Option(
        None, "--table", "-t", help="CSV amplitude table instead of captures."
    ),
    config_file: Path = ConfigOption,
    set_: list[str] = SetOption,
    out: Path = typer.Option(
        None, "--out", "-o", help="Output folder (default: a new run folder)."
    ),
) -> None:
    """Calibrate, resample, window and decompose; write the feature table."""
    from .capture.dataset import DatasetManifest
    from .pipeline import build_dataset, build_table_dataset

    config = _load_config(config_file, set_)
    if dataset is not None and table is not None:
        raise typer.BadParameter("give either --dataset or --table, not both")
    with _reported_errors():
        if table is not None:
            if not table.is_file():
                raise FileNotFoundError(f"amplitude table not found: {table}")
            features, details = build_table_dataset(config, table)
            inputs = [table]
        else:
            manifest_path = dataset or Path(config.dataset)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"dataset manifest not found: {manifest_path}. Write one for your captures "
                    "(see configs/datasets/) or run 'wifi-presence-sensing synthesize' first."
                )
            manifest = DatasetManifest.from_yaml(manifest_path, data_root=config.data_root)
            missing = [r.path for r in manifest.recordings if not r.path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} capture(s) listed in {manifest_path} are missing, "
                    f"e.g. {missing[0]}; check the paths or set data_root"
                )
            features, details = build_dataset(manifest, config)
            inputs = [manifest_path, *[r.path for r in manifest.recordings]]
    folder = _run_folder(config, "prepare", out)
    destination = features.to_csv(folder / "features.csv")
    _finish(folder, "prepare", config, inputs, details)
    typer.echo(
        f"{features.n_windows} windows x {len(features.feature_names)} features, "
        f"labels {sorted(set(features.labels.tolist()))}"
    )
    typer.echo(f"feature table: {destination}")


@app.command()
def decompose(
    capture: Path = typer.Argument(..., help="Capture file (.dat)."),
    stream: str = typer.Option("tx0rx0", "--stream", help="Antenna pair, e.g. tx0rx1."),
    config_file: Path = ConfigOption,
    set_: list[str] = SetOption,
    out: Path = typer.Option(
        None, "--out", "-o", help="Output folder (default: a new run folder)."
    ),
) -> None:
    """Decompose one antenna pair of a capture and write its IMFs, statistics and figure."""
    from .capture.dataset import RecordingEntry
    from .decompose import decompose as run_decomposition
    from .features.modal_statistics import mode_statistics
    from .features.subcarrier_profile import reduce_subcarriers
    from .pipeline import calibrate_trace, load_trace, trace_to_series
    from .report.figures import plot_decomposition, save_figure
    from .report.manifest import write_json

    config = _load_config(config_file, set_)
    if not capture.is_file():
        raise typer.BadParameter(f"capture not found: {capture}")
    with _reported_errors():
        trace, _, _ = load_trace(capture, config)
        entry = RecordingEntry(path=capture.resolve(), label="unlabelled", session=capture.stem)
        series_list, _ = trace_to_series(calibrate_trace(trace, config), entry, config)
    if not series_list:
        raise typer.BadParameter("no contiguous segment long enough; relax resample.max_gap_s")
    series = max(series_list, key=lambda s: s.n_samples)
    if stream not in series.stream_names:
        raise typer.BadParameter(f"stream {stream!r} not in {list(series.stream_names)}")
    decomposition = config.decomposition
    with _reported_errors():
        signal = reduce_subcarriers(
            series.amplitude_db[:, series.stream_names.index(stream), :],
            config.features.stream_reduction,
        )
        result = run_decomposition(
            signal,
            decomposition.method,
            n_ensemble=decomposition.n_ensemble,
            noise_scale=decomposition.noise_scale,
            max_modes=decomposition.max_modes,
            stopping=decomposition.stopping_criterion(),
            max_sift_iterations=decomposition.max_sift_iterations,
            seed=config.seed,
            paired_noise=decomposition.paired_noise,
        )
    folder = _run_folder(config, "decompose", out)
    time = series.start_s + np.arange(signal.size) / series.sampling_rate_hz
    frame = pd.DataFrame({"time_s": time, "signal_db": signal})
    for k, mode in enumerate(result.imfs):
        frame[f"imf{k + 1}"] = mode
    frame["residual"] = result.residual
    frame.to_csv(folder / "imfs.csv", index=False)
    stats = mode_statistics(result.imfs, result.residual, series.sampling_rate_hz)
    write_json(folder / "mode_statistics.json", stats)
    save_figure(
        plot_decomposition(
            signal,
            result.imfs,
            result.residual,
            series.sampling_rate_hz,
            title=f"{capture.name} {stream}: {decomposition.method.upper()}",
        ),
        folder / "decomposition",
        config.output.figure_format,
    )
    _finish(folder, "decompose", config, [capture], {"stream": stream, "n_modes": len(result.imfs)})
    typer.echo(
        f"{len(result.imfs)} IMF(s) of {signal.size} samples at {series.sampling_rate_hz:g} Hz"
    )
    typer.echo(f"written to {folder}")


def _read_features(path: Path):  # type: ignore[no-untyped-def]
    from .features.extraction import FeatureTable

    if not path.is_file():
        raise typer.BadParameter(f"feature table not found: {path}")
    with _reported_errors():
        return FeatureTable.read_csv(path)


@app.command()
def train(
    features: Path = typer.Argument(..., help="Feature table written by 'prepare'."),
    model_out: Path = typer.Option(
        None,
        "--model-out",
        "-m",
        help="Model file to write (default: occupancy_model.joblib under output.root).",
    ),
    config_file: Path = ConfigOption,
    set_: list[str] = SetOption,
) -> None:
    """Fit the occupancy classifier on the whole feature table and save it."""
    from .occupancy.classifier import OccupancyClassifier, save_classifier

    config = _load_config(config_file, set_)
    table = _read_features(features)
    occupancy = config.occupancy
    with _reported_errors():
        classifier = OccupancyClassifier(
            model=occupancy.model,
            feature_groups=occupancy.feature_groups,
            regularization=occupancy.regularization,
            n_estimators=occupancy.n_estimators,
            seed=config.seed,
        ).fit(table.feature_frame(), table.labels)
    if model_out is None:
        model_out = Path(config.output.root) / "occupancy_model.joblib"
    save_classifier(
        classifier,
        model_out,
        metadata={
            "config": config.to_dict(),
            "classes": classifier.classes_.tolist(),
            "n_training_windows": table.n_windows,
            "features": classifier.selected_features_,
        },
    )
    typer.echo(
        f"trained {occupancy.model} on {table.n_windows} windows, "
        f"{len(classifier.selected_features_ or [])} features -> {model_out}"
    )


@app.command()
def evaluate(
    features: Path = typer.Argument(..., help="Feature table written by 'prepare'."),
    config_file: Path = ConfigOption,
    set_: list[str] = SetOption,
    out: Path = typer.Option(
        None, "--out", "-o", help="Output folder (default: a new run folder)."
    ),
) -> None:
    """Grouped cross-validation of the classifier against the baselines."""
    from .pipeline import evaluate_occupancy, evaluated_windows, occupancy_summary
    from .report.figures import confusion_matrix_table, plot_confusion_matrix, save_figure
    from .report.manifest import write_json

    config = _load_config(config_file, set_)
    table = _read_features(features)
    with _reported_errors():
        # Block grouping drops windows that straddle a block boundary; the
        # out-of-fold predictions line up with the windows that remain.
        evaluated = evaluated_windows(table, config.evaluation.group_by, config)
        reports = evaluate_occupancy(evaluated, config)
    folder = _run_folder(config, "evaluate", out)
    summary = occupancy_summary(reports)
    summary.to_csv(folder / "summary.csv", index=False)
    write_json(folder / "metrics.json", {name: r.as_dict() for name, r in reports.items()})
    for name, report in reports.items():
        stem = folder / f"confusion_{name}"
        save_figure(
            plot_confusion_matrix(
                report.pooled.confusion, report.pooled.labels, title=f"{name} (pooled out-of-fold)"
            ),
            stem,
            config.output.figure_format,
        )
        confusion_matrix_table(report.pooled.confusion, report.pooled.labels).to_csv(
            f"{stem}.csv", index=False
        )
    predictions = pd.DataFrame(
        {
            "recording": evaluated.recordings,
            "window_start_s": evaluated.window_start_s,
            "label": evaluated.labels,
            **{f"predicted_{name}": r.predictions for name, r in reports.items()},
        }
    )
    predictions.to_csv(folder / "out_of_fold_predictions.csv", index=False)
    _finish(
        folder,
        "evaluate",
        config,
        [features],
        {
            "group_by": config.evaluation.group_by,
            "n_windows": evaluated.n_windows,
            "n_windows_dropped_at_block_boundaries": table.n_windows - evaluated.n_windows,
        },
    )
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        typer.echo(summary.round(3).to_string(index=False))
    typer.echo(f"written to {folder}")


@app.command()
def predict(
    model: Path = typer.Argument(..., help="Classifier saved by 'train'."),
    features: Path = typer.Argument(..., help="Feature table written by 'prepare'."),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Prediction CSV file to write (default: predictions.csv under the output.root "
        "the model was trained with).",
    ),
) -> None:
    """Apply a saved classifier to every window of a feature table."""
    from .occupancy.classifier import load_classifier

    configure_logging("INFO")
    table = _read_features(features)
    with _reported_errors():
        classifier, metadata = load_classifier(model)
        frame = table.feature_frame()
        labels = classifier.predict(frame)
        probabilities = classifier.predict_proba(frame)
    result = pd.DataFrame(
        {
            "recording": table.recordings,
            "window_start_s": table.window_start_s,
            "predicted": labels,
            **{f"p_{cls}": probabilities[:, i] for i, cls in enumerate(classifier.classes_)},
        }
    )
    if out is None:
        root = metadata.get("config", {}).get("output", {}).get("root", OutputConfig().root)
        out = Path(root) / "predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    counts = pd.Series(labels).value_counts().to_dict()
    typer.echo(f"{len(labels)} windows classified {counts} -> {out}")


@app.command()
def localize(
    features: Path = typer.Argument(..., help="Feature table with positions."),
    config_file: Path = ConfigOption,
    set_: list[str] = SetOption,
    out: Path = typer.Option(
        None, "--out", "-o", help="Output folder (default: a new run folder)."
    ),
) -> None:
    """Compare the received-power, raw-response and modal-response locators."""
    from .pipeline import evaluate_localization, evaluated_windows
    from .report.figures import (
        error_cdf_table,
        error_curve_table,
        plot_error_cdfs,
        save_figure,
    )
    from .report.manifest import write_json

    config = _load_config(config_file, set_)
    table = _read_features(features)
    with _reported_errors():
        evaluated = evaluated_windows(table, config.localization.group_by, config)
        benchmark = evaluate_localization(evaluated, config)
    folder = _run_folder(config, "localize", out)
    error_cdf_table(benchmark.results).to_csv(folder / "error_cdf.csv", index=False)
    error_curve_table(benchmark.results).to_csv(folder / "error_curve.csv", index=False)
    pd.DataFrame(
        {
            "recording": evaluated.recordings,
            "window_start_s": evaluated.window_start_s,
            "true_x": evaluated.positions[:, 0],
            "true_y": evaluated.positions[:, 1],
            **{
                f"{name}_{axis}": estimate[:, index]
                for name, estimate in benchmark.predictions.items()
                for index, axis in enumerate(("x", "y"))
            },
        }
    ).to_csv(folder / "estimates.csv", index=False)
    write_json(folder / "metrics.json", benchmark.as_dict())
    save_figure(
        plot_error_cdfs(benchmark.results), folder / "error_cdf", config.output.figure_format
    )
    _finish(
        folder,
        "localize",
        config,
        [features],
        {
            "group_by": config.localization.group_by,
            "n_folds": benchmark.n_folds,
            "n_windows": evaluated.n_windows,
            "n_windows_dropped_at_block_boundaries": table.n_windows - evaluated.n_windows,
        },
    )
    for name, result in benchmark.results.items():
        summary = result.summary
        typer.echo(
            f"{name:>15}: median {summary.median:.2f} m, mean {summary.mean:.2f} m, "
            f"p90 {summary.p90:.2f} m, "
            f"CDF@{result.thresholds_m[0]:g} m {result.cdf[0]:.3f}, n={result.n_samples}"
        )
    typer.echo(f"written to {folder}")


def main() -> None:
    """Entry point of the ``wifi-presence-sensing`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
