"""Loader for amplitude tables exported as plain CSV.

Some experiments keep CSI as a headerless numeric table instead of raw
captures: one row per packet, a few leading label columns, then the dB
amplitude of every (antenna pair, subcarrier) cell.  An unfilled cell in such
an export is written as the 16-bit sentinel ``65535``, which parses as an
ordinary number.  A single one raises a stream's mean by tens of dB and its
standard deviation by orders of magnitude.

:func:`load_amplitude_table` checks the schema, sets sentinels, empty cells and
values outside a plausible dB range to NaN, and logs how many entries and rows
were masked.  No row is dropped; :meth:`AmplitudeTable.quality_report` breaks
the count down per stream.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

#: Accepted column layouts of the amplitude block.
TABLE_LAYOUTS: tuple[str, ...] = ("stream_major", "subcarrier_major")


class TableSchemaError(ValueError):
    """Raised when a table does not match the declared schema."""


@dataclass(frozen=True, eq=False)
class AmplitudeTable:
    """A validated amplitude table.

    Attributes
    ----------
    labels : numpy.ndarray
        Integer label columns, shaped ``(n_rows, n_label_columns)``.
    amplitude_db : numpy.ndarray
        Amplitudes shaped ``(n_rows, n_streams, n_subcarriers)``; masked
        entries are NaN.
    masked : numpy.ndarray
        Boolean mask of the same shape, True where a value was rejected.
    n_sentinel : int
        Entries rejected because they equal a sentinel value.
    n_out_of_range : int
        Entries rejected because they fall outside the valid dB range.
    source : str
        File name.
    """

    labels: np.ndarray
    amplitude_db: np.ndarray
    masked: np.ndarray
    n_sentinel: int
    n_out_of_range: int
    source: str = ""

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return int(self.labels.shape[0])

    def quality_report(self) -> dict[str, object]:
        """Summarize what was masked, for logs and run manifests."""
        n_entries = int(self.masked.size)
        n_masked = int(self.masked.sum())
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_entries": n_entries,
            "n_masked": n_masked,
            "masked_fraction": n_masked / n_entries if n_entries else 0.0,
            "rows_with_masked_entries": int(self.masked.any(axis=(1, 2)).sum()),
            "masked_per_stream": self.masked.sum(axis=(0, 2)).astype(int).tolist(),
            "n_sentinel": self.n_sentinel,
            "n_out_of_range": self.n_out_of_range,
        }


def load_amplitude_table(
    path: str | Path,
    *,
    n_label_columns: int = 2,
    n_streams: int = 6,
    n_subcarriers: int = 30,
    sentinel_values: Sequence[float] = (65535.0,),
    valid_range_db: tuple[float, float] = (-40.0, 80.0),
    layout: str = "stream_major",
    delimiter: str = ",",
) -> AmplitudeTable:
    """Read and validate a headerless amplitude table.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file.
    n_label_columns : int, optional
        Leading integer label columns.
    n_streams, n_subcarriers : int, optional
        Shape of the amplitude block; the file must have exactly
        ``n_label_columns + n_streams * n_subcarriers`` columns.
    sentinel_values : sequence of float, optional
        Values that mean "missing".
    valid_range_db : (float, float), optional
        Amplitudes outside this closed interval are masked.
    layout : {"stream_major", "subcarrier_major"}, optional
        ``stream_major``: all subcarriers of stream 0, then stream 1, ...
    delimiter : str, optional
        Field separator.

    Returns
    -------
    AmplitudeTable
        The validated table.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TableSchemaError
        If the column count is wrong, a cell is not numeric, or a label is
        not an integer.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"amplitude table not found: {source}")
    if layout not in TABLE_LAYOUTS:
        raise TableSchemaError(f"layout={layout!r} is not one of {TABLE_LAYOUTS}")
    low, high = valid_range_db
    if not low < high:
        raise TableSchemaError(f"valid_range_db must be increasing, got {valid_range_db}")

    frame = pd.read_csv(source, header=None, sep=delimiter, dtype=str, keep_default_na=False)
    expected_columns = n_label_columns + n_streams * n_subcarriers
    if frame.shape[1] != expected_columns:
        raise TableSchemaError(
            f"{source.name} has {frame.shape[1]} columns; the schema expects "
            f"{n_label_columns} label + {n_streams}x{n_subcarriers} amplitude = {expected_columns}"
        )
    stripped = frame.apply(lambda column: column.str.strip())
    numeric = stripped.apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() & (stripped != "")
    if bad.to_numpy().any():
        row, column = np.argwhere(bad.to_numpy())[0]
        raise TableSchemaError(
            f"{source.name}: non-numeric cell {stripped.iat[row, column]!r} at row {row}, "
            f"column {column}"
        )

    values = numeric.to_numpy(dtype=np.float64)
    label_values = values[:, :n_label_columns]
    if np.isnan(label_values).any() or not np.allclose(label_values, np.rint(label_values)):
        raise TableSchemaError(f"{source.name}: label columns must hold integers in every row")
    labels = np.rint(label_values).astype(np.int64)

    block = values[:, n_label_columns:]
    sentinel = np.isin(block, np.asarray(sentinel_values, dtype=np.float64))
    empty = np.isnan(block)
    out_of_range = ~sentinel & ~empty & ((block < low) | (block > high))
    masked = sentinel | empty | out_of_range
    cleaned = np.where(masked, np.nan, block)

    if layout == "stream_major":
        amplitude = cleaned.reshape(-1, n_streams, n_subcarriers)
        mask3 = masked.reshape(-1, n_streams, n_subcarriers)
    else:
        amplitude = cleaned.reshape(-1, n_subcarriers, n_streams).transpose(0, 2, 1)
        mask3 = masked.reshape(-1, n_subcarriers, n_streams).transpose(0, 2, 1)

    table = AmplitudeTable(
        labels=labels,
        amplitude_db=np.ascontiguousarray(amplitude),
        masked=np.ascontiguousarray(mask3),
        n_sentinel=int(sentinel.sum()),
        n_out_of_range=int(out_of_range.sum()),
        source=source.name,
    )
    report = table.quality_report()
    if report["n_masked"]:
        _LOGGER.warning(
            "%s: masked %d of %d entries (%.3f%%) in %d row(s): %d sentinel, %d out of range",
            source.name,
            report["n_masked"],
            report["n_entries"],
            100.0 * float(report["masked_fraction"]),  # type: ignore[arg-type]
            report["rows_with_masked_entries"],
            table.n_sentinel,
            table.n_out_of_range,
        )
    return table


def contiguous_runs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label maximal runs of identical consecutive rows.

    Rows recorded back to back under the same condition are strongly
    correlated in time; their run is the natural cross-validation group.

    Parameters
    ----------
    keys : numpy.ndarray
        Shaped ``(n_rows,)`` or ``(n_rows, n_keys)``.

    Returns
    -------
    run_ids : numpy.ndarray
        Run index of every row, starting at 0.
    starts : numpy.ndarray
        First row of each run.
    lengths : numpy.ndarray
        Number of rows in each run.

    Examples
    --------
    >>> ids, starts, lengths = contiguous_runs(np.array([[2, 1], [2, 1], [3, 1], [2, 1]]))
    >>> ids.tolist(), starts.tolist(), lengths.tolist()
    ([0, 0, 1, 2], [0, 2, 3], [2, 1, 1])
    """
    array = np.asarray(keys)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    change = np.any(array[1:] != array[:-1], axis=1)
    run_ids = np.concatenate([[0], np.cumsum(change)]).astype(np.int64)
    starts = np.concatenate([[0], np.flatnonzero(change) + 1]).astype(np.int64)
    lengths = np.diff(np.concatenate([starts, [array.shape[0]]])).astype(np.int64)
    return run_ids, starts, lengths


def label_balance(labels: np.ndarray) -> dict[tuple[int, ...], int]:
    """Count rows per label combination.

    Examples
    --------
    >>> label_balance(np.array([[2, 1], [2, 1], [3, 1]]))
    {(2, 1): 2, (3, 1): 1}
    """
    combos, counts = np.unique(np.asarray(labels), axis=0, return_counts=True)
    return {tuple(int(v) for v in combo): int(n) for combo, n in zip(combos, counts, strict=True)}


def fill_masked_entries(amplitude_db: np.ndarray, run_ids: np.ndarray) -> np.ndarray:
    """Fill masked (NaN) amplitudes by interpolation along time within each run.

    Interpolation never crosses a run boundary, so one condition never leaks
    into another.  Cells that are missing for an entire run fall back to the
    table-wide median of that (stream, subcarrier) cell; since that median
    draws on other runs, the fallback is logged with the number of values it
    filled.

    Parameters
    ----------
    amplitude_db : numpy.ndarray
        Shaped ``(n_rows, n_streams, n_subcarriers)`` with NaN for masked.
    run_ids : numpy.ndarray
        Run index of every row (see :func:`contiguous_runs`).

    Returns
    -------
    numpy.ndarray
        Copy without NaN.

    Raises
    ------
    ValueError
        If a cell is NaN in every row of the table.
    """
    shape = amplitude_db.shape
    flat = pd.DataFrame(amplitude_db.reshape(shape[0], -1))
    if not flat.isna().to_numpy().any():
        return amplitude_db.copy()
    filled = flat.groupby(np.asarray(run_ids), sort=False, group_keys=False).transform(
        lambda column: column.interpolate(limit_direction="both")
    )
    n_unresolved = int(filled.isna().to_numpy().sum())
    if n_unresolved:
        medians = flat.median(axis=0, skipna=True)
        if medians.isna().any():
            raise ValueError("at least one (stream, subcarrier) cell is masked in every row")
        _LOGGER.warning(
            "%d value(s) of cells masked throughout their run were filled with the "
            "table-wide median of the cell, which draws on other runs",
            n_unresolved,
        )
        filled = filled.fillna(medians)
    return filled.to_numpy(dtype=np.float64).reshape(shape)
