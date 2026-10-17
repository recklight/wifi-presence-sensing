"""From capture files to evaluation results.

Which recordings go in comes from the dataset manifest, and how each stage
runs comes from the configuration:

.. code-block:: text

    capture file ──read──> snapshots ──stack──> ChannelTrace
        ──spatial mapping──> ──noise model──> ──absolute scaling──> amplitude (dB)
        ──uniform resampling per contiguous segment──> AmplitudeSeries
        ──per-window power, response, profile and EEMD modal features──> FeatureTable
        ──grouped CV──> occupancy metrics  /  locator error CDFs

Every stage returns, besides its data, a small report of what it kept,
dropped, masked or replaced; the CLI writes those reports into the run
manifest.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrate.link_quality import packet_effective_snr_db
from .calibrate.noise_model import NoiseEstimate, NoiseModel
from .calibrate.resample import resample_uniform
from .calibrate.scaling import (
    RSSI_OFFSET_DB,
    amplitude_db,
    to_calibrated_csi,
    total_received_power_dbm,
)
from .calibrate.spatial_mapping import (
    mapping_from_phases,
    standard_mapping_matrix,
    undo_spatial_mapping,
)
from .capture.beamform_report import ReadStatistics, read_beamform_file
from .capture.dataset import DatasetManifest, RecordingEntry
from .capture.subcarriers import require_single_bandwidth
from .capture.tabular_reader import contiguous_runs, fill_masked_entries, load_amplitude_table
from .capture.timeline import sampling_diagnostics
from .capture.trace import ChannelTrace, StackingReport, stack_snapshots
from .config import Config
from .evaluate.crossval import CrossValidationReport, compare_classifiers
from .features.extraction import AmplitudeSeries, FeatureTable, build_feature_table, stream_names
from .localize.benchmark import LocalizationBenchmark, benchmark_locators, default_locators
from .occupancy.classifier import candidate_classifiers

_LOGGER = logging.getLogger(__name__)

_CHAIN_NAMES = ("chain_a_dbm", "chain_b_dbm", "chain_c_dbm")


@dataclass(eq=False)
class CalibratedTrace:
    """A trace after spatial-mapping inversion, noise modelling and scaling.

    Attributes
    ----------
    trace : ChannelTrace
        The stacked trace (sorted by time).
    amplitude_db : numpy.ndarray
        ``20 log10 |H|`` of the absolutely scaled CSI, ``(n, n_tx, n_rx, n_sc)``.
    power_dbm : numpy.ndarray
        Received power of every reporting chain and in total, ``(n, n_power)``.
    power_names : tuple of str
        Names of the power columns.
    noise : NoiseEstimate
        Cleaned noise floor.
    effective_snr_db : dict
        Effective SNR per packet for each modulation.
    """

    trace: ChannelTrace
    amplitude_db: np.ndarray
    power_dbm: np.ndarray
    power_names: tuple[str, ...]
    noise: NoiseEstimate
    effective_snr_db: dict[str, np.ndarray]


def spatial_mapping_for(config: Config, n_tx: int) -> np.ndarray | None:
    """The configured spatial mapping matrix for ``n_tx`` chains (``None``: no inversion)."""
    calibration = config.calibration
    if calibration.spatial_mapping == "none" or n_tx == 1:
        return None
    if calibration.spatial_mapping == "phases":
        matrix = mapping_from_phases(np.asarray(calibration.mapping_phases_rad, dtype=np.float64))
        if matrix.shape != (n_tx, n_tx):
            raise ValueError(
                f"calibration.mapping_phases_rad is {matrix.shape[0]}x{matrix.shape[1]} "
                f"but the trace has {n_tx} transmit chains"
            )
        return matrix
    return standard_mapping_matrix(n_tx, calibration.spatial_mapping)


def load_trace(
    path: str | Path, config: Config
) -> tuple[ChannelTrace, ReadStatistics, StackingReport]:
    """Read a capture and stack it into a time-sorted, homogeneous trace."""
    source = Path(path)
    snapshots, statistics = read_beamform_file(
        source, apply_permutation=config.capture.apply_antenna_permutation
    )
    if not snapshots:
        raise ValueError(f"{source.name}: no decodable beamforming report")
    trace, stacking = stack_snapshots(
        snapshots, tx_policy=config.capture.tx_policy, source=source.name
    )
    order = np.argsort(trace.timestamps_s, kind="stable")
    if np.any(order != np.arange(order.size)):
        trace = trace.select(order)
    return trace, statistics, stacking


def calibrate_trace(trace: ChannelTrace, config: Config) -> CalibratedTrace:
    """Undo the spatial mapping, clean the noise floor and put the CSI on an absolute scale."""
    csi = trace.csi
    mapping = spatial_mapping_for(config, trace.n_tx)
    if mapping is not None:
        csi = undo_spatial_mapping(csi, mapping)

    calibration = config.calibration
    noise = NoiseModel(
        strategy=calibration.noise_strategy,
        sentinel_dbm=calibration.noise_sentinel_dbm,
        fallback_dbm=calibration.noise_fallback_dbm,
        threshold=calibration.noise_threshold,
        window=calibration.noise_window,
        plausible_range_dbm=calibration.noise_plausible_range_dbm,
    ).estimate(trace.noise_dbm)

    absolute = to_calibrated_csi(csi, trace.rssi_db, trace.agc_db, reference="absolute")
    snr_units = to_calibrated_csi(
        csi,
        trace.rssi_db,
        trace.agc_db,
        noise_dbm=noise.noise_dbm,
        reference="noise",
        include_quantization_noise=calibration.include_quantization_noise,
    )
    # Per-chain columns only for chains that report in every packet; the total
    # sums whatever chains reported in each packet, exactly like the absolute
    # scaling above, so "total_dbm" and the calibrated CSI share one reference.
    present = np.all(trace.rssi_db != 0, axis=0)
    chains = trace.rssi_db[:, present] - RSSI_OFFSET_DB - trace.agc_db[:, None]
    total = total_received_power_dbm(trace.rssi_db, trace.agc_db)
    power = np.column_stack([chains, total])
    names = (*[name for name, keep in zip(_CHAIN_NAMES, present, strict=True) if keep], "total_dbm")

    finite = np.all(np.isfinite(snr_units), axis=(1, 2, 3))
    effective = packet_effective_snr_db(snr_units[finite]) if finite.any() else {}
    return CalibratedTrace(
        trace=trace,
        amplitude_db=amplitude_db(absolute),
        power_dbm=power,
        power_names=names,
        noise=noise,
        effective_snr_db=effective,
    )


def trace_to_series(
    calibrated: CalibratedTrace, entry: RecordingEntry, config: Config
) -> tuple[list[AmplitudeSeries], dict[str, Any]]:
    """Resample a calibrated trace onto uniform grids, one series per contiguous segment."""
    trace = calibrated.trace
    n = trace.n_packets
    amplitude = calibrated.amplitude_db.reshape(n, trace.n_tx * trace.n_rx, -1)
    valid = np.all(np.isfinite(amplitude), axis=(1, 2)) & np.all(
        np.isfinite(calibrated.power_dbm), axis=1
    )
    if not valid.all():
        _LOGGER.warning(
            "%s: %d packet(s) without a finite calibration dropped", entry.name, int((~valid).sum())
        )
    width = amplitude.shape[1] * amplitude.shape[2]
    joined = np.concatenate([amplitude.reshape(n, -1), calibrated.power_dbm], axis=1)[valid]
    segments = resample_uniform(
        trace.timestamps_s[valid],
        joined,
        config.resample.rate_hz,
        max_gap_s=config.resample.max_gap_s,
        method=config.resample.method,
        min_duration_s=config.resample.min_segment_s,
    )
    names = stream_names(trace.n_tx, trace.n_rx)
    series = [
        AmplitudeSeries(
            amplitude_db=segment.values[:, :width].reshape(segment.n_samples, len(names), -1),
            sampling_rate_hz=segment.rate_hz,
            stream_names=names,
            label=entry.label,
            recording=entry.name,
            session=entry.session,
            start_s=segment.start_s,
            position=entry.position,
            power_dbm=segment.values[:, width:],
            power_names=calibrated.power_names,
        )
        for segment in segments
    ]
    report = {
        "n_segments": len(segments),
        "resampled_seconds": float(sum(s.duration_s for s in segments)),
        "interpolated_fraction": float(
            np.mean([s.interpolated_fraction for s in segments]) if segments else 0.0
        ),
        "methods": sorted({s.method for s in segments}),
    }
    return series, report


def inspect_capture(path: str | Path, config: Config) -> tuple[dict[str, Any], ChannelTrace]:
    """Diagnostics of one capture: what was read, its radio setup, sampling and noise.

    Returns
    -------
    report : dict
        JSON-ready diagnostics.
    trace : ChannelTrace
        The stacked, time-sorted trace the diagnostics describe.
    """
    trace, statistics, stacking = load_trace(path, config)
    calibrated = calibrate_trace(trace, config)
    diagnostics = sampling_diagnostics(trace.timestamps_s, max_gap_s=config.resample.max_gap_s)
    report = {
        "file": Path(path).name,
        "path": Path(path).as_posix(),
        "read": statistics.as_dict(),
        "stacking": asdict(stacking),
        "bandwidth_mhz": trace.bandwidth_mhz,
        "shape": {"n_packets": trace.n_packets, "n_tx": trace.n_tx, "n_rx": trace.n_rx},
        "sampling": diagnostics.as_dict(),
        "sampling_limits": diagnostics.describe_limits(),
        "agc_db_range": [float(trace.agc_db.min()), float(trace.agc_db.max())],
        "noise": {
            "strategy": calibrated.noise.strategy,
            "replaced_fraction": calibrated.noise.replaced_fraction,
            "reported_range_dbm": [float(trace.noise_dbm.min()), float(trace.noise_dbm.max())],
        },
        "median_effective_snr_db": {
            name: float(np.median(values)) for name, values in calibrated.effective_snr_db.items()
        },
    }
    return report, trace


def recording_series(
    entry: RecordingEntry, config: Config
) -> tuple[list[AmplitudeSeries], dict[str, Any]]:
    """Read, calibrate and resample one recording of a manifest."""
    trace, statistics, stacking = load_trace(entry.path, config)
    calibrated = calibrate_trace(trace, config)
    series, resampling = trace_to_series(calibrated, entry, config)
    diagnostics = sampling_diagnostics(trace.timestamps_s, max_gap_s=config.resample.max_gap_s)
    report = {
        "recording": entry.name,
        "path": entry.path.as_posix(),
        "label": entry.label,
        "bandwidth_mhz": trace.bandwidth_mhz,
        "read": statistics.as_dict(),
        "stacking": asdict(stacking),
        "sampling": diagnostics.as_dict(),
        "noise_replaced_fraction": calibrated.noise.replaced_fraction,
        "resampling": resampling,
    }
    if diagnostics.mean_rate_hz < config.resample.rate_hz:
        _LOGGER.warning(
            "%s: long-run packet rate %.3g Hz is below the resampling rate %.3g Hz; "
            "the grid holds interpolated rather than measured detail",
            entry.name,
            diagnostics.mean_rate_hz,
            config.resample.rate_hz,
        )
    return series, report


def build_dataset(manifest: DatasetManifest, config: Config) -> tuple[FeatureTable, dict[str, Any]]:
    """Feature table of every recording of a manifest.

    Recordings at another bandwidth than ``capture.bandwidth_mhz`` are skipped;
    without that setting, a dataset mixing bandwidths is rejected.
    """
    all_series: list[AmplitudeSeries] = []
    reports: list[dict[str, Any]] = []
    bandwidths: list[int] = []
    for entry in manifest.recordings:
        series, report = recording_series(entry, config)
        wanted = config.capture.bandwidth_mhz
        if wanted is not None and report["bandwidth_mhz"] != wanted:
            _LOGGER.warning(
                "%s skipped: %d MHz capture, the analysis is restricted to %d MHz",
                entry.name,
                report["bandwidth_mhz"],
                wanted,
            )
            report["skipped"] = "bandwidth"
            reports.append(report)
            continue
        bandwidths.append(int(report["bandwidth_mhz"]))
        reports.append(report)
        all_series.extend(series)
    require_single_bandwidth(bandwidths)
    table = build_feature_table(
        all_series, decomposition=config.decomposition, features=config.features, seed=config.seed
    )
    return table, {"dataset": manifest.name, "recordings": reports, "n_windows": table.n_windows}


def table_series(
    config: Config, path: str | Path | None = None
) -> tuple[list[AmplitudeSeries], dict[str, Any]]:
    """Amplitude series from a CSV amplitude table, one per contiguous run of identical labels."""
    table_config = config.table
    table = load_amplitude_table(
        path or table_config.path,
        n_label_columns=table_config.n_label_columns,
        n_streams=table_config.n_streams,
        n_subcarriers=table_config.n_subcarriers,
        sentinel_values=table_config.sentinel_values,
        valid_range_db=table_config.valid_range_db,
        layout=table_config.layout,
    )
    run_ids, starts, lengths = contiguous_runs(table.labels)
    filled = fill_masked_entries(table.amplitude_db, run_ids)
    # A table does not say which physical chains each stream came from, so the
    # streams get neutral names rather than guessed transmit/receive indices.
    names = tuple(f"stream{i}" for i in range(table_config.n_streams))
    series = []
    for run, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        labels = table.labels[start]
        tag = "-".join(str(int(v)) for v in labels)
        series.append(
            AmplitudeSeries(
                amplitude_db=filled[start : start + length],
                sampling_rate_hz=table_config.nominal_rate_hz,
                stream_names=names,
                label=str(int(labels[table_config.label_column])),
                recording=f"run{run:03d}_{tag}",
                session=f"run{run:03d}_{tag}",
            )
        )
    report = table.quality_report()
    report["n_runs"] = int(starts.size)
    report["run_lengths"] = sorted({int(v) for v in lengths})
    return series, report


def build_table_dataset(
    config: Config, path: str | Path | None = None
) -> tuple[FeatureTable, dict[str, Any]]:
    """Feature table from a CSV amplitude table."""
    series, report = table_series(config, path)
    table = build_feature_table(
        series, decomposition=config.decomposition, features=config.features, seed=config.seed
    )
    report["n_windows"] = table.n_windows
    return table, report


def _param_grids(config: Config) -> dict[str, dict[str, list[Any]]]:
    if not config.evaluation.nested_search:
        return {}
    if config.occupancy.model == "random_forest":
        grid: dict[str, list[Any]] = {"n_estimators": [100, 300, 600]}
    elif config.occupancy.model == "majority":
        return {}
    else:
        grid = {"regularization": [0.1, 1.0, 10.0]}
    return {"modal": grid, "raw_response": grid}


def evaluated_windows(table: FeatureTable, group_by: str, config: Config) -> FeatureTable:
    """The windows a grouped evaluation actually runs on.

    Grouping by ``block`` first drops the windows that straddle a block
    boundary, since they share samples with two blocks; every other grouping
    uses the table unchanged.  Row ``i`` of an evaluation result (out-of-fold
    predictions, locator estimates) refers to row ``i`` of this table, which
    can be shorter than the table that was passed in.

    The purge uses the window length recorded in the table; a configuration
    whose ``features.window_s`` differs only triggers a warning.

    Parameters
    ----------
    table : FeatureTable
        The full feature table.
    group_by : str
        ``recording``, ``session`` or ``block``.
    config : Config
        Supplies ``features.block_s`` (and ``features.window_s`` for a table
        that does not record its window length).

    Returns
    -------
    FeatureTable
        The evaluated windows (the input itself unless ``group_by`` is ``block``).
    """
    if group_by != "block":
        return table
    window_s = config.features.window_s
    if table.window_s is not None:
        if not np.isclose(table.window_s, window_s):
            _LOGGER.warning(
                "the feature table holds %.4g s windows; the block purge uses that length, "
                "not features.window_s = %.4g s from the configuration",
                table.window_s,
                window_s,
            )
        window_s = table.window_s
    return table.with_block_purge(window_s, config.features.block_s)


def evaluate_occupancy(table: FeatureTable, config: Config) -> dict[str, CrossValidationReport]:
    """Grouped cross-validation of the proposed classifier and its baselines.

    The reports cover the windows of
    ``evaluated_windows(table, config.evaluation.group_by, config)``.
    """
    occupancy = config.occupancy
    candidates = candidate_classifiers(
        model=occupancy.model,
        feature_groups=occupancy.feature_groups,
        regularization=occupancy.regularization,
        n_estimators=occupancy.n_estimators,
        seed=config.seed,
    )
    table = evaluated_windows(table, config.evaluation.group_by, config)
    labels = set(table.labels.tolist())
    positive = occupancy.positive_label if occupancy.positive_label in labels else None
    return compare_classifiers(
        candidates,
        table.feature_frame(),
        table.labels,
        table.groups(config.evaluation.group_by),
        n_splits=config.evaluation.n_splits,
        seed=config.seed,
        positive_label=positive,
        param_grids=_param_grids(config),
        inner_splits=config.evaluation.inner_splits,
    )


def occupancy_summary(reports: dict[str, CrossValidationReport]) -> pd.DataFrame:
    """One row per candidate: pooled metrics and the fold mean and spread of UAR."""
    rows = []
    for name, report in reports.items():
        headline = report.pooled.headline()
        folds = report.fold_summary().get("uar", {})
        rows.append(
            {
                "candidate": name,
                "uar": headline["uar"],
                "uar_fold_mean": folds.get("mean"),
                "uar_fold_std": folds.get("std"),
                "accuracy": headline["accuracy"],
                "macro_f1": headline["macro_f1"],
                "sensitivity": headline.get("sensitivity"),
                "specificity": headline.get("specificity"),
                "roc_auc": headline["roc_auc"],
                "n_samples": report.pooled.n_samples,
            }
        )
    return pd.DataFrame(rows)


def evaluate_localization(table: FeatureTable, config: Config) -> LocalizationBenchmark:
    """Grouped cross-validated comparison of the three fingerprint locators.

    The estimates cover the windows of
    ``evaluated_windows(table, config.localization.group_by, config)``.
    """
    localization = config.localization
    table = evaluated_windows(table, localization.group_by, config)
    return benchmark_locators(
        table,
        locators=default_locators(localization.method, localization.n_neighbors),
        group_by=localization.group_by,
        n_splits=config.evaluation.n_splits,
        thresholds_m=localization.cdf_thresholds_m,
        n_bootstrap=localization.n_bootstrap,
        confidence=localization.confidence,
        seed=config.seed,
    )
