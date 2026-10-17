from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from wifi_presence_sensing.capture.tabular_reader import (
    TableSchemaError,
    contiguous_runs,
    fill_masked_entries,
    label_balance,
    load_amplitude_table,
)
from wifi_presence_sensing.synthetic import synthetic_amplitude_table


def write_table(path: Path, table: np.ndarray) -> Path:
    np.savetxt(path, table, delimiter=",", fmt="%.4f")
    return path


def test_sentinels_are_masked_and_reported(tmp_path: Path, caplog):
    raw = synthetic_amplitude_table(rows_per_block=20, n_sentinels=15, seed=1)
    expected = int(np.sum(raw[:, 2:] == 65535.0))
    path = write_table(tmp_path / "table.csv", raw)
    with caplog.at_level(logging.WARNING):
        table = load_amplitude_table(path)
    report = table.quality_report()
    assert table.n_sentinel == expected
    assert report["n_masked"] == expected
    assert sum(report["masked_per_stream"]) == expected
    assert np.isnan(table.amplitude_db).sum() == expected
    assert np.nanmax(table.amplitude_db) < 100.0
    assert "masked" in caplog.text


def test_out_of_range_values_are_masked(tmp_path: Path):
    raw = synthetic_amplitude_table(rows_per_block=10, n_sentinels=0, seed=2)
    raw[3, 10] = 500.0
    raw[4, 11] = -300.0
    table = load_amplitude_table(write_table(tmp_path / "t.csv", raw))
    assert table.n_out_of_range == 2
    assert table.masked.sum() == 2


def test_layout_is_stream_major_by_default(tmp_path: Path):
    raw = np.zeros((2, 2 + 6 * 30))
    raw[:, :2] = [[1, 1], [1, 1]]
    raw[:, 2 + 30 : 2 + 60] = 7.0  # the second block of 30 columns is stream 1
    table = load_amplitude_table(write_table(tmp_path / "t.csv", raw))
    assert np.all(table.amplitude_db[:, 1, :] == 7.0)
    assert np.all(table.amplitude_db[:, 0, :] == 0.0)
    alt = load_amplitude_table(tmp_path / "t.csv", layout="subcarrier_major")
    assert alt.amplitude_db.shape == (2, 6, 30)


def test_schema_errors_are_explicit(tmp_path: Path):
    bad_columns = tmp_path / "cols.csv"
    np.savetxt(bad_columns, np.ones((3, 50)), delimiter=",")
    with pytest.raises(TableSchemaError, match="columns"):
        load_amplitude_table(bad_columns)

    text = tmp_path / "text.csv"
    rows = [",".join(["1", "2"] + ["3.0"] * 180) for _ in range(3)]
    rows[1] = rows[1].replace("3.0", "abc", 1)
    text.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(TableSchemaError, match="non-numeric"):
        load_amplitude_table(text)

    fractional = tmp_path / "labels.csv"
    data = np.ones((2, 182))
    data[0, 0] = 2.5
    np.savetxt(fractional, data, delimiter=",")
    with pytest.raises(TableSchemaError, match="integers"):
        load_amplitude_table(fractional)
    with pytest.raises(FileNotFoundError):
        load_amplitude_table(tmp_path / "missing.csv")


def test_runs_and_balance_of_a_block_layout():
    raw = synthetic_amplitude_table(rows_per_block=25, n_sentinels=0)
    labels = raw[:, :2].astype(int)
    run_ids, starts, lengths = contiguous_runs(labels)
    assert starts.size == 9
    assert set(lengths.tolist()) == {25}
    assert run_ids[-1] == 8
    balance = label_balance(labels)
    assert len(balance) == 9 and set(balance.values()) == {25}


def test_fill_never_crosses_a_run_boundary():
    amplitude = np.zeros((6, 1, 2))
    amplitude[:3] = 1.0
    amplitude[3:] = 10.0
    amplitude[2, 0, 0] = np.nan  # last row of run 0
    amplitude[3, 0, 1] = np.nan  # first row of run 1
    run_ids = np.array([0, 0, 0, 1, 1, 1])
    filled = fill_masked_entries(amplitude, run_ids)
    assert filled[2, 0, 0] == 1.0
    assert filled[3, 0, 1] == 10.0
    everything_missing = np.full((4, 1, 1), np.nan)
    with pytest.raises(ValueError):
        fill_masked_entries(everything_missing, np.zeros(4, dtype=int))


def test_whole_run_gaps_fall_back_to_the_table_median_and_say_so(caplog):
    amplitude = np.zeros((6, 1, 2))
    amplitude[:3] = 1.0
    amplitude[3:] = 3.0
    amplitude[3:, 0, 0] = np.nan  # this cell is masked for the whole of run 1
    run_ids = np.array([0, 0, 0, 1, 1, 1])
    with caplog.at_level(logging.WARNING):
        filled = fill_masked_entries(amplitude, run_ids)
    np.testing.assert_array_equal(filled[3:, 0, 0], 1.0)  # median of the cell over the table
    np.testing.assert_array_equal(filled[3:, 0, 1], 3.0)
    assert "draws on other runs" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fill_masked_entries(np.ones((4, 1, 2)), np.zeros(4, dtype=int))
    assert caplog.text == ""
