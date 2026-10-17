"""End-to-end demonstration on simulated captures only.

Runs the whole pipeline without any measurement data:

1. simulate beamforming-report captures (empty and occupied rooms in several
   sessions; a person standing at grid positions) and write them as binary
   capture files with dataset manifests;
2. inspect one capture: packets, timestamp wrap, sampling limits, noise
   cleaning, effective SNR;
3. calibrate, resample, window and decompose (EEMD) every recording into a
   feature table;
4. occupancy: grouped cross-validation of the modal-feature classifier against
   the raw-amplitude variant, a single mean-amplitude feature and a majority
   vote;
5. localization: grouped cross-validation of the received-power, raw-response
   and modal-response locators, with error-distance CDFs and bootstrap bands.

Everything is written under ``--output`` (default ``examples/output``) with a
manifest per stage.  The simulated room responds strongly to a person, so the
scores say that the software works, not how well real rooms can be sensed.

Usage::

    python examples/end_to_end_synthetic.py
    python examples/end_to_end_synthetic.py --output /tmp/wifi-presence-demo --sessions 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import wifi_presence_sensing
except ImportError:  # running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import wifi_presence_sensing

from wifi_presence_sensing.capture import DatasetManifest
from wifi_presence_sensing.config import Config
from wifi_presence_sensing.logging_utils import configure_logging
from wifi_presence_sensing.pipeline import (
    build_dataset,
    evaluate_localization,
    evaluate_occupancy,
    inspect_capture,
    occupancy_summary,
)
from wifi_presence_sensing.report import (
    build_manifest,
    confusion_matrix_table,
    error_cdf_table,
    error_curve_table,
    plot_confusion_matrix,
    plot_error_cdfs,
    save_figure,
    write_json,
    write_manifest,
)
from wifi_presence_sensing.synthetic import write_synthetic_dataset

LOGGER = logging.getLogger("end_to_end_synthetic")


def demo_config(seed: int) -> Config:
    """Small but sound settings for short simulated recordings."""
    return Config().override(
        {
            "seed": seed,
            "resample.rate_hz": 10.0,
            "resample.min_segment_s": 6.4,
            "decomposition.method": "eemd",
            "decomposition.n_ensemble": 8,
            "decomposition.max_modes": 3,
            "features.window_s": 6.4,
            "features.hop_s": 3.2,
            "features.block_s": 12.8,
            "evaluation.n_splits": 3,
            "evaluation.group_by": "session",
            "localization.n_bootstrap": 500,
        }
    )


def run_occupancy(output: Path, config: Config, sessions: int) -> None:
    manifest_path = write_synthetic_dataset(
        output / "captures" / "occupancy",
        kind="occupancy",
        n_sessions=sessions,
        duration_s=40.0,
        seed=config.seed,
    )
    manifest = DatasetManifest.from_yaml(manifest_path)

    report, _ = inspect_capture(manifest.recordings[1].path, config)
    LOGGER.info(
        "inspected %s: %d packets, %.1f s, radio setups %s",
        report["file"],
        report["shape"]["n_packets"],
        report["sampling"]["duration_s"],
        report["stacking"]["radio_setups"],
    )
    for line in report["sampling_limits"]:
        LOGGER.info("  %s", line)
    LOGGER.info(
        "  noise values replaced: %.1f%%; median effective SNR (QPSK): %.1f dB",
        100 * report["noise"]["replaced_fraction"],
        report["median_effective_snr_db"]["qpsk"],
    )

    table, details = build_dataset(manifest, config)
    folder = output / "occupancy"
    table.to_csv(folder / "features.csv")
    reports = evaluate_occupancy(table, config)
    summary = occupancy_summary(reports)
    summary.to_csv(folder / "summary.csv", index=False)
    write_json(folder / "metrics.json", {name: r.as_dict() for name, r in reports.items()})
    pooled = reports["modal"].pooled
    save_figure(
        plot_confusion_matrix(
            pooled.confusion, pooled.labels, title="modal features, pooled out-of-fold"
        ),
        folder / "confusion_modal",
    )
    confusion_matrix_table(pooled.confusion, pooled.labels).to_csv(
        folder / "confusion_modal.csv", index=False
    )
    write_manifest(
        folder / "manifest.json",
        build_manifest(
            command="example: occupancy",
            config=config.to_dict(),
            inputs=[manifest_path, *[r.path for r in manifest.recordings]],
            seed=config.seed,
            extra=details,
        ),
    )
    LOGGER.info("occupancy (%d windows, grouped by session):", table.n_windows)
    for row in summary.itertuples(index=False):
        LOGGER.info(
            "  %-15s UAR %.3f (folds %.3f +/- %.3f)  sensitivity %.3f  specificity %.3f  AUC %s",
            row.candidate,
            row.uar,
            row.uar_fold_mean,
            row.uar_fold_std,
            row.sensitivity,
            row.specificity,
            "n/a" if row.roc_auc is None else f"{row.roc_auc:.3f}",
        )


def run_localization(output: Path, config: Config, sessions: int) -> None:
    manifest_path = write_synthetic_dataset(
        output / "captures" / "localization",
        kind="localization",
        n_sessions=sessions,
        duration_s=16.0,
        grid_shape=(3, 3),
        seed=config.seed + 1,
    )
    manifest = DatasetManifest.from_yaml(manifest_path)
    table, details = build_dataset(manifest, config)
    folder = output / "localization"
    table.to_csv(folder / "features.csv")
    benchmark = evaluate_localization(table, config)
    error_cdf_table(benchmark.results).to_csv(folder / "error_cdf.csv", index=False)
    error_curve_table(benchmark.results).to_csv(folder / "error_curve.csv", index=False)
    write_json(folder / "metrics.json", benchmark.as_dict())
    save_figure(plot_error_cdfs(benchmark.results), folder / "error_cdf")
    write_manifest(
        folder / "manifest.json",
        build_manifest(
            command="example: localization",
            config=config.to_dict(),
            inputs=[manifest_path, *[r.path for r in manifest.recordings]],
            seed=config.seed,
            extra=details,
        ),
    )
    LOGGER.info("localization (%d windows, %d folds):", table.n_windows, benchmark.n_folds)
    for name, result in benchmark.results.items():
        summary = result.summary
        LOGGER.info(
            "  %-15s median %.2f m  mean %.2f m  p90 %.2f m  CDF@1m %.3f [%.3f, %.3f]",
            name,
            summary.median,
            summary.mean,
            summary.p90,
            result.cdf[0],
            result.ci_low[0],
            result.ci_high[0],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Folder for captures, tables, figures and manifests.",
    )
    parser.add_argument("--sessions", type=int, default=4, help="Recording sessions to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Master seed.")
    args = parser.parse_args(argv)

    configure_logging("INFO")
    config = demo_config(args.seed)
    LOGGER.info(
        "wifi_presence_sensing %s, writing to %s", wifi_presence_sensing.__version__, args.output
    )
    run_occupancy(args.output, config, args.sessions)
    run_localization(args.output, config, max(2, args.sessions - 1))
    LOGGER.info("done: see %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
