"""Figures written as image files next to the numbers they show.

Every figure is built on a standalone :class:`matplotlib.figure.Figure` (no
global pyplot state, no display needed) and has a companion table with the
plotted values (:func:`error_cdf_table`, :func:`error_curve_table`,
:func:`confusion_matrix_table`, :func:`packet_interval_table`; the CLI writes
them as CSV), so a result can be diffed and re-plotted without opening an
image.

Series use a fixed blue/orange/green order that stays distinguishable for
common colour-vision deficiencies, and each series also gets its own marker.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..evaluate.error_cdf import ErrorCdfResult

_LOGGER = logging.getLogger(__name__)

#: Line colours for up to three series, always in this order: blue, orange, green.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
#: One marker per series, so the curves can be told apart without relying on colour.
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^")
#: Light-to-dark blues for the confusion-matrix cells.
SEQUENTIAL_RAMP: tuple[str, ...] = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"


def _new_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), dpi=150, facecolor=_SURFACE, layout="constrained")
    return figure


def _style_axes(axes) -> None:  # type: ignore[no-untyped-def]
    axes.set_facecolor(_SURFACE)
    axes.grid(True, color=_GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_TEXT_SECONDARY)
        axes.spines[side].set_linewidth(0.6)
    axes.tick_params(colors=_TEXT_SECONDARY, labelsize=8)
    axes.xaxis.label.set_color(_TEXT)
    axes.yaxis.label.set_color(_TEXT)
    axes.title.set_color(_TEXT)


def save_figure(figure: Figure, path_stem: str | Path, figure_format: str = "png") -> Path:
    """Save a figure as ``<path_stem>.<format>``.

    The extension is appended, never substituted, so a stem that itself holds
    a dot (a capture named ``2024.05.03_room``) keeps its full name.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = save_figure(_new_figure(1.0, 1.0), Path(folder) / "2024.05.03_room_intervals")
    ...     path.name
    '2024.05.03_room_intervals.png'
    """
    stem = Path(path_stem)
    destination = stem.with_name(f"{stem.name}.{figure_format}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor())
    _LOGGER.info("figure written to %s", destination)
    return destination


def plot_decomposition(
    signal: np.ndarray,
    imfs: np.ndarray,
    residual: np.ndarray,
    sampling_rate_hz: float,
    *,
    title: str = "",
) -> Figure:
    """Signal, each IMF and the residual on a shared time axis, one row each."""
    n_rows = imfs.shape[0] + 2
    time = np.arange(np.asarray(signal).size) / sampling_rate_hz
    figure = _new_figure(7.0, 1.2 * n_rows + 0.6)
    axes = figure.subplots(n_rows, 1, sharex=True)
    rows = [("signal", signal), *[(f"IMF {k + 1}", m) for k, m in enumerate(imfs)]]
    rows.append(("residual", residual))
    for ax, (name, values) in zip(np.atleast_1d(axes), rows, strict=True):
        _style_axes(ax)
        ax.plot(time, values, color=SERIES_COLORS[0], linewidth=1.0)
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    if title:
        figure.suptitle(title, color=_TEXT, fontsize=10)
    return figure


def plot_error_cdfs(results: Mapping[str, ErrorCdfResult], *, title: str = "") -> Figure:
    """Full empirical error CDF of each locator, with the tabulated thresholds and band.

    At most three series are drawn so the colours stay distinguishable.
    """
    if len(results) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} curves per figure, got {len(results)}")
    figure = _new_figure(6.0, 4.0)
    ax = figure.subplots()
    _style_axes(ax)
    for index, (name, result) in enumerate(results.items()):
        color, marker = SERIES_COLORS[index], SERIES_MARKERS[index]
        errors, probability = result.empirical_curve()
        ax.step(errors, probability, where="post", color=color, linewidth=1.5, alpha=0.9)
        has_band = np.all(np.isfinite(result.ci_low))
        if has_band:
            ax.errorbar(
                result.thresholds_m,
                result.cdf,
                yerr=[result.cdf - result.ci_low, result.ci_high - result.cdf],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
            )
        ax.plot(
            result.thresholds_m,
            result.cdf,
            linestyle="none",
            marker=marker,
            markersize=6,
            color=color,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.0,
            label=(
                f"{name} (median {result.summary.median:.2f} m, "
                f"{result.fraction_beyond_max:.0%} > {result.thresholds_m[-1]:g} m)"
            ),
        )
    ax.set_xlabel("error distance (m)")
    ax.set_ylabel("cumulative probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0.0)
    n_values = {r.n_samples for r in results.values()}
    ax.set_title(
        title or f"Localization error CDF (n = {', '.join(map(str, sorted(n_values)))})",
        fontsize=10,
    )
    ax.legend(fontsize=7, frameon=False, labelcolor=_TEXT, loc="lower right")
    return figure


def error_cdf_table(results: Mapping[str, ErrorCdfResult]) -> pd.DataFrame:
    """Long-format table of the tabulated CDFs (one row per locator and threshold)."""
    rows = []
    for name, result in results.items():
        for index, threshold in enumerate(result.thresholds_m):
            rows.append(
                {
                    "locator": name,
                    "threshold_m": float(threshold),
                    "cdf": float(result.cdf[index]),
                    "ci_low": float(result.ci_low[index]),
                    "ci_high": float(result.ci_high[index]),
                    "n_samples": result.n_samples,
                    "fraction_beyond_max": result.fraction_beyond_max,
                }
            )
    return pd.DataFrame(rows)


#: Columns of :func:`error_curve_table`.
ERROR_CURVE_COLUMNS: tuple[str, ...] = ("locator", "error_m", "cumulative_probability")


def error_curve_table(results: Mapping[str, ErrorCdfResult]) -> pd.DataFrame:
    """Long-format table of the full empirical CDFs: every error distance of every locator.

    One row per evaluated window, sorted by error: the raw error vector behind
    each plotted curve.  With it a curve can be re-plotted, tabulated at other
    thresholds or compared byte for byte without rerunning the benchmark.

    Parameters
    ----------
    results : mapping
        ``{locator name: ErrorCdfResult}``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`ERROR_CURVE_COLUMNS`.

    Examples
    --------
    >>> from wifi_presence_sensing.evaluate.error_cdf import error_cdf_from_errors
    >>> curve = error_cdf_from_errors(np.array([2.0, 0.5]), n_bootstrap=0)
    >>> error_curve_table({"modal_response": curve}).values.tolist()
    [['modal_response', 0.5, 0.5], ['modal_response', 2.0, 1.0]]
    """
    frames = []
    for name, result in results.items():
        errors, probability = result.empirical_curve()
        frames.append(
            pd.DataFrame(
                {
                    "locator": [name] * errors.size,
                    "error_m": errors,
                    "cumulative_probability": probability,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(ERROR_CURVE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def confusion_matrix_table(confusion: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix as a table: one row per true class, one count column per prediction.

    Parameters
    ----------
    confusion : numpy.ndarray
        Square matrix, rows true classes, columns predicted classes.
    labels : sequence of str
        Class order of both axes.

    Returns
    -------
    pandas.DataFrame
        Column ``true_label`` followed by ``predicted_<label>`` counts.

    Examples
    --------
    >>> table = confusion_matrix_table(np.array([[9, 1], [2, 8]]), ["empty", "occupied"])
    >>> list(table.columns)
    ['true_label', 'predicted_empty', 'predicted_occupied']
    >>> table["predicted_empty"].tolist(), table["predicted_occupied"].tolist()
    ([9, 2], [1, 8])
    """
    matrix = np.asarray(confusion)
    names = [str(label) for label in labels]
    if matrix.shape != (len(names), len(names)):
        raise ValueError(f"confusion matrix {matrix.shape} does not match {len(names)} labels")
    table = pd.DataFrame(matrix, columns=[f"predicted_{name}" for name in names])
    table.insert(0, "true_label", names)
    return table


def packet_interval_table(times_s: np.ndarray) -> pd.DataFrame:
    """Intervals between consecutive packets, the data behind :func:`plot_sampling_intervals`.

    Examples
    --------
    >>> packet_interval_table(np.array([0.0, 0.5, 2.0])).values.tolist()
    [[0.0, 0.5], [0.5, 1.5]]
    """
    times = np.asarray(times_s, dtype=np.float64)
    return pd.DataFrame({"start_s": times[:-1], "interval_s": np.diff(times)})


def plot_confusion_matrix(
    confusion: np.ndarray, labels: Sequence[str], *, title: str = ""
) -> Figure:
    """Confusion matrix with counts and row-normalized recall in every cell."""
    from matplotlib.colors import LinearSegmentedColormap

    matrix = np.asarray(confusion, dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(totals > 0, matrix / totals, 0.0)
    figure = _new_figure(1.2 * len(labels) + 2.2, 1.0 * len(labels) + 1.6)
    ax = figure.subplots()
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_RAMP)
    image = ax.imshow(recall, cmap=cmap, vmin=0.0, vmax=1.0)
    for (row, column), count in np.ndenumerate(matrix):
        ink = "#ffffff" if recall[row, column] > 0.55 else _TEXT
        ax.text(
            column,
            row,
            f"{int(count)}\n{recall[row, column]:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color=ink,
        )
    ax.set_xticks(range(len(labels)), labels=list(labels), fontsize=8, color=_TEXT_SECONDARY)
    ax.set_yticks(range(len(labels)), labels=list(labels), fontsize=8, color=_TEXT_SECONDARY)
    ax.set_xlabel("predicted", color=_TEXT)
    ax.set_ylabel("true", color=_TEXT)
    if title:
        ax.set_title(title, fontsize=10, color=_TEXT)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.046)
    colorbar.set_label("row recall", color=_TEXT_SECONDARY, fontsize=8)
    colorbar.ax.tick_params(labelsize=7, colors=_TEXT_SECONDARY)
    return figure


def plot_sampling_intervals(times_s: np.ndarray, *, title: str = "") -> Figure:
    """Histogram of packet intervals on a log axis: bursts and pauses at a glance."""
    intervals = np.diff(np.asarray(times_s, dtype=np.float64))
    intervals = intervals[intervals > 0]
    figure = _new_figure(6.0, 3.2)
    ax = figure.subplots()
    _style_axes(ax)
    if intervals.size:
        bins = np.logspace(np.log10(intervals.min()), np.log10(intervals.max()) + 1e-9, 40)
        ax.hist(intervals, bins=bins, color=SERIES_COLORS[0], edgecolor=_SURFACE, linewidth=0.8)
        ax.set_xscale("log")
    ax.set_xlabel("packet interval (s)")
    ax.set_ylabel("count")
    ax.set_title(title or "Packet interval distribution", fontsize=10)
    return figure
