"""Reporting: figures with companion tables, run manifests."""

from __future__ import annotations

from .figures import (
    ERROR_CURVE_COLUMNS,
    SERIES_COLORS,
    confusion_matrix_table,
    error_cdf_table,
    error_curve_table,
    packet_interval_table,
    plot_confusion_matrix,
    plot_decomposition,
    plot_error_cdfs,
    plot_sampling_intervals,
    save_figure,
)
from .manifest import (
    array_sha256,
    build_manifest,
    file_sha256,
    git_commit,
    run_directory,
    write_json,
    write_manifest,
)

__all__ = [
    "ERROR_CURVE_COLUMNS",
    "SERIES_COLORS",
    "array_sha256",
    "build_manifest",
    "confusion_matrix_table",
    "error_cdf_table",
    "error_curve_table",
    "file_sha256",
    "git_commit",
    "packet_interval_table",
    "plot_confusion_matrix",
    "plot_decomposition",
    "plot_error_cdfs",
    "plot_sampling_intervals",
    "run_directory",
    "save_figure",
    "write_json",
    "write_manifest",
]
