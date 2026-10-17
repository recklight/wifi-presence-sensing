"""Grouped splits, classification metrics and error CDFs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wifi_presence_sensing.evaluate import (
    SplitError,
    assert_disjoint_groups,
    classification_metrics,
    compare_classifiers,
    cross_validate_grouped,
    error_cdf_from_errors,
    error_distance_cdf,
    find_duplicate_curves,
    grouped_folds,
    grouped_stratified_folds,
    localization_errors,
    sensitivity_specificity,
    summarize_errors,
)
from wifi_presence_sensing.occupancy import OccupancyClassifier
from wifi_presence_sensing.report import (
    ERROR_CURVE_COLUMNS,
    error_cdf_table,
    error_curve_table,
)


def blocks(n_groups_per_class: int, rows_per_group: int, classes=("empty", "occupied")):
    labels, groups = [], []
    for c in classes:
        for g in range(n_groups_per_class):
            labels += [c] * rows_per_group
            groups += [f"{c}-{g}"] * rows_per_group
    return np.array(labels), np.array(groups)


@pytest.mark.parametrize("seed", range(5))
def test_stratified_group_folds_never_share_a_group(seed: int):
    labels, groups = blocks(6, 10, classes=("a", "b", "c"))
    folds = grouped_stratified_folds(labels, groups, n_splits=3, seed=seed)
    assert len(folds) == 3
    covered = np.concatenate([test for _, test in folds])
    assert sorted(covered.tolist()) == list(range(labels.size))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert set(labels[test]) == {"a", "b", "c"}


def test_too_few_groups_is_explained():
    labels, groups = blocks(2, 10)
    with pytest.raises(SplitError, match="fewer groups than n_splits"):
        grouped_stratified_folds(labels, groups, n_splits=5, seed=0)
    with pytest.raises(SplitError, match="two classes"):
        grouped_stratified_folds(
            np.array(["a"] * 4), np.array(["g1", "g2", "g3", "g4"]), n_splits=2, seed=0
        )
    with pytest.raises(SplitError):
        assert_disjoint_groups(np.array([0]), np.array([1]), np.array(["x", "x"]))


def test_group_folds_for_regression_reduce_the_split_count():
    groups = np.repeat(["s1", "s2", "s3"], 4)
    folds = grouped_folds(groups, n_splits=5)
    assert len(folds) == 3
    with pytest.raises(SplitError):
        grouped_folds(np.array(["only"] * 5), n_splits=2)


def test_random_splits_would_leak_adjacent_windows():
    """Why grouping matters: windows of one recording are near-duplicates."""
    rng = np.random.default_rng(0)
    labels, groups = blocks(4, 30)
    offsets = {g: rng.normal(0, 3, 5) for g in np.unique(groups)}  # recording fingerprint
    X = np.stack([offsets[g] + 0.05 * rng.normal(size=5) for g in groups])
    frame = pd.DataFrame(X, columns=[f"modal__f{i}" for i in range(5)])
    grouped = cross_validate_grouped(
        OccupancyClassifier(model="random_forest", n_estimators=50, feature_groups=("modal",)),
        frame,
        labels,
        groups,
        n_splits=4,
        seed=0,
    )
    # Labels are unrelated to the features here, so UAR on a grouped split stays near chance,
    # whereas a split that shares recordings would score close to perfect.
    assert grouped.pooled.uar < 0.8


def test_metrics_on_a_known_confusion():
    truth = ["occupied"] * 10 + ["empty"] * 10
    pred = ["occupied"] * 8 + ["empty"] * 2 + ["empty"] * 9 + ["occupied"]
    scores = np.column_stack([np.r_[np.full(10, 0.3), np.full(10, 0.8)], np.zeros(20)])
    scores[:, 1] = 1 - scores[:, 0]
    m = classification_metrics(
        truth, pred, labels=["empty", "occupied"], scores=scores, positive_label="occupied"
    )
    assert m.accuracy == pytest.approx(0.85)
    assert m.uar == pytest.approx(0.85)
    assert m.sensitivity["occupied"] == pytest.approx(0.8)
    assert m.specificity["occupied"] == pytest.approx(0.9)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.confusion.tolist() == [[9, 1], [2, 8]]
    headline = m.headline()
    assert headline["sensitivity"] == pytest.approx(0.8)
    assert m.as_dict()["confusion"] == [[9, 1], [2, 8]]


def test_uar_is_insensitive_to_class_imbalance():
    truth = ["empty"] * 95 + ["occupied"] * 5
    always_empty = ["empty"] * 100
    m = classification_metrics(truth, always_empty)
    assert m.accuracy == pytest.approx(0.95)
    assert m.uar == pytest.approx(0.5)


def test_multiclass_metrics_and_auc():
    rng = np.random.default_rng(1)
    truth = np.repeat(["a", "b", "c"], 20)
    scores = rng.dirichlet(np.ones(3), size=60)
    scores[np.arange(60), np.repeat([0, 1, 2], 20)] += 2.0
    pred = np.array(["a", "b", "c"])[scores.argmax(axis=1)]
    m = classification_metrics(truth, pred, labels=["a", "b", "c"], scores=scores)
    assert m.roc_auc == pytest.approx(1.0)
    sens, spec = sensitivity_specificity(m.confusion)
    assert np.all(sens == 1.0) and np.all(spec == 1.0)
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_compare_classifiers_uses_identical_folds():
    rng = np.random.default_rng(3)
    labels, groups = blocks(5, 12)
    signal = (labels == "occupied").astype(float)
    frame = pd.DataFrame(
        {
            "modal__a": signal * 2 + rng.normal(0, 0.5, labels.size),
            "response__mean_db": rng.normal(0, 1, labels.size),
        }
    )
    reports = compare_classifiers(
        {
            "modal": OccupancyClassifier(feature_groups=("modal",)),
            "mean": OccupancyClassifier(feature_columns=("response__mean_db",)),
            "majority": OccupancyClassifier(model="majority", feature_groups=None),
        },
        frame,
        labels,
        groups,
        n_splits=5,
        seed=0,
        positive_label="occupied",
    )
    assert [r.fold_groups for r in reports.values()].count(reports["modal"].fold_groups) == 3
    assert reports["modal"].pooled.uar > 0.9
    assert reports["mean"].pooled.uar < 0.75
    assert reports["majority"].pooled.uar == pytest.approx(0.5)
    summary = reports["modal"].fold_summary()
    assert summary["uar"]["mean"] > 0.9
    assert "pooled" in reports["modal"].as_dict()


def test_nested_search_reports_chosen_parameters():
    rng = np.random.default_rng(4)
    labels, groups = blocks(6, 8)
    frame = pd.DataFrame({"modal__a": (labels == "occupied") * 1.5 + rng.normal(0, 1, labels.size)})
    report = cross_validate_grouped(
        OccupancyClassifier(feature_groups=("modal",)),
        frame,
        labels,
        groups,
        n_splits=3,
        seed=0,
        param_grid={"regularization": [0.01, 1.0]},
        inner_splits=2,
    )
    assert len(report.best_params) == 3
    assert all(p["regularization"] in (0.01, 1.0) for p in report.best_params)


def test_error_cdf_bookkeeping():
    errors = np.r_[np.full(30, 0.5), np.full(50, 2.5), np.full(20, 12.0)]
    result = error_cdf_from_errors(errors, thresholds_m=range(1, 9), n_bootstrap=300, seed=1)
    assert result.n_samples == 100
    assert result.cdf.tolist() == [0.3, 0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    assert result.fraction_beyond_max == pytest.approx(0.2)
    assert np.all(result.ci_low <= result.cdf) and np.all(result.cdf <= result.ci_high)
    assert result.ci_high[0] - result.ci_low[0] > 0.05
    again = error_cdf_from_errors(errors, thresholds_m=range(1, 9), n_bootstrap=300, seed=1)
    np.testing.assert_array_equal(result.ci_low, again.ci_low)
    assert result.summary.median == pytest.approx(2.5)
    payload = result.as_dict()
    assert payload["n_samples"] == 100 and "errors" not in payload


def test_error_cdf_from_positions_and_validation():
    truth = np.zeros((3, 2))
    estimate = np.array([[3.0, 4.0], [0.0, 1.0], [0.0, 0.0]])
    assert localization_errors(truth, estimate).tolist() == [5.0, 1.0, 0.0]
    result = error_distance_cdf(truth, estimate, thresholds_m=[1, 5], n_bootstrap=0)
    assert result.cdf.tolist() == pytest.approx([2 / 3, 1.0])
    assert np.all(np.isnan(result.ci_low))
    with pytest.raises(ValueError):
        error_cdf_from_errors(np.array([1.0]), thresholds_m=[2, 1])
    with pytest.raises(ValueError):
        summarize_errors(np.array([-1.0]))


def test_identical_curves_are_flagged(caplog):
    base = error_cdf_from_errors(np.array([1.0, 2.0, 3.0]), n_bootstrap=0)
    copy = error_cdf_from_errors(np.array([3.0, 1.0, 2.0]), n_bootstrap=0)
    other = error_cdf_from_errors(np.array([1.0, 2.0, 4.0]), n_bootstrap=0)
    curves = {"received_power": base, "received_power_rerun": copy, "modal_response": other}
    assert find_duplicate_curves(curves) == [["received_power", "received_power_rerun"]]
    assert "identical" in caplog.text


def test_error_curve_table_holds_every_error_behind_the_tabulated_cdf(rng):
    results = {
        "received_power": error_cdf_from_errors(rng.gamma(2.0, 1.5, size=40), n_bootstrap=0),
        "modal_response": error_cdf_from_errors(rng.gamma(2.0, 0.8, size=40), n_bootstrap=0),
    }
    curve = error_curve_table(results)
    assert tuple(curve.columns) == ERROR_CURVE_COLUMNS
    assert len(curve) == 80
    tabulated = error_cdf_table(results)
    for name in results:
        errors = curve.loc[curve["locator"] == name, "error_m"].to_numpy()
        probability = curve.loc[curve["locator"] == name, "cumulative_probability"].to_numpy()
        assert np.all(np.diff(errors) >= 0)
        assert probability[-1] == pytest.approx(1.0)
        # Every tabulated CDF value can be recomputed from the saved raw errors.
        rows = tabulated[tabulated["locator"] == name]
        for threshold, value in zip(rows["threshold_m"], rows["cdf"], strict=True):
            assert np.mean(errors <= threshold) == pytest.approx(value)
    assert error_curve_table({}).empty
