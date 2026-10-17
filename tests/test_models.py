"""Occupancy classifier and fingerprint locators on synthetic feature matrices."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from wifi_presence_sensing.features import FeatureTable
from wifi_presence_sensing.localize import (
    ModalResponseLocator,
    RadioMapLocator,
    RawResponseLocator,
    ReceivedPowerLocator,
    benchmark_locators,
)
from wifi_presence_sensing.occupancy import (
    OCCUPANCY_MODELS,
    OccupancyClassifier,
    candidate_classifiers,
    load_classifier,
    save_classifier,
)


def labelled_frame(rng: np.random.Generator, n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    labels = np.repeat(["empty", "occupied"], n // 2)
    shift = (labels == "occupied").astype(float)
    frame = pd.DataFrame(
        {
            "modal__a": 2.0 * shift + rng.normal(0, 0.5, n),
            "modal__b": -1.5 * shift + rng.normal(0, 0.5, n),
            "profile__c": rng.normal(0, 1, n),
            "response__mean_db": rng.normal(0, 1, n),
        }
    )
    return frame, labels


@pytest.mark.parametrize("model", OCCUPANCY_MODELS)
def test_every_model_fits_and_predicts(model: str, rng):
    frame, labels = labelled_frame(rng)
    classifier = OccupancyClassifier(model=model, n_estimators=30, seed=1).fit(frame, labels)
    probabilities = classifier.predict_proba(frame)
    assert probabilities.shape == (len(labels), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    accuracy = classifier.score(frame, labels)
    assert accuracy > (0.4 if model == "majority" else 0.9)


def test_svm_probabilities_raise_no_deprecation_warning(rng):
    # scikit-learn 1.9 deprecates SVC(probability=True) with a FutureWarning.
    frame, labels = labelled_frame(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        classifier = OccupancyClassifier(model="svm", seed=1).fit(frame, labels)
        probabilities = classifier.predict_proba(frame)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


@pytest.mark.parametrize("per_class", [2, 3, 4])
def test_svm_fits_with_fewer_than_five_windows_per_class(per_class: int, rng):
    # The Platt calibration uses as many folds as the smallest class allows, up to 5.
    frame, labels = labelled_frame(rng, n=2 * per_class)
    classifier = OccupancyClassifier(model="svm", seed=1).fit(frame, labels)
    assert classifier.predict_proba(frame).shape == (2 * per_class, 2)
    with pytest.raises(ValueError):
        OccupancyClassifier(model="svm").fit(frame.iloc[: per_class + 1], labels[: per_class + 1])


def test_feature_selection_by_group_and_by_column(rng):
    frame, labels = labelled_frame(rng)
    by_group = OccupancyClassifier(feature_groups=("modal",)).fit(frame, labels)
    assert by_group.selected_features_ == ["modal__a", "modal__b"]
    by_column = OccupancyClassifier(feature_columns=("response__mean_db",)).fit(frame, labels)
    assert by_column.selected_features_ == ["response__mean_db"]
    with pytest.raises(ValueError, match="no column matches"):
        OccupancyClassifier(feature_groups=("power",)).fit(frame, labels)
    with pytest.raises(ValueError, match="lacks"):
        by_group.predict(frame.drop(columns=["modal__b"]))
    plain = OccupancyClassifier().fit(frame.to_numpy(), labels)
    assert plain.selected_features_ is None


def test_classifier_input_validation(rng):
    frame, labels = labelled_frame(rng)
    with pytest.raises(ValueError, match="two classes"):
        OccupancyClassifier().fit(frame, ["empty"] * len(labels))
    broken = frame.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        OccupancyClassifier().fit(broken, labels)
    with pytest.raises(ValueError):
        OccupancyClassifier(model="xgboost").fit(frame, labels)


def test_classifier_is_clonable_and_persists(tmp_path: Path, rng):
    frame, labels = labelled_frame(rng)
    candidates = candidate_classifiers(seed=3)
    assert set(candidates) == {"modal", "raw_response", "majority", "mean_amplitude"}
    fitted = clone(candidates["modal"]).fit(frame, labels)
    path = save_classifier(fitted, tmp_path / "model.joblib", metadata={"note": "test"})
    loaded, metadata = load_classifier(path)
    np.testing.assert_array_equal(loaded.predict(frame), fitted.predict(frame))
    assert metadata == {"note": "test"}
    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / "absent.joblib")


def grid_fingerprints(rng: np.random.Generator, sessions: int = 3, per_point: int = 4):
    positions = np.array([[x, y] for x in (1.0, 4.0, 7.0) for y in (1.0, 5.0)])
    rows, xy, session_ids = [], [], []
    for s in range(sessions):
        for point in positions:
            for _ in range(per_point):
                base = np.r_[point, point[0] * point[1]]
                rows.append(
                    {
                        "power__total_dbm_mean": -50 - point[0] + rng.normal(0, 3.0),
                        "response__k0_db": base[0] + rng.normal(0, 0.8),
                        "response__k1_db": base[1] + rng.normal(0, 0.8),
                        "modal__m1": base[0] + rng.normal(0, 0.1),
                        "modal__m2": base[1] + rng.normal(0, 0.1),
                        "profile__p": base[2] / 10 + rng.normal(0, 0.1),
                    }
                )
                xy.append(point)
                session_ids.append(f"session{s}")
    return pd.DataFrame(rows), np.array(xy), np.array(session_ids)


@pytest.mark.parametrize("method", ["knn", "probabilistic"])
def test_locators_recover_reference_positions(method: str, rng):
    frame, xy, _ = grid_fingerprints(rng)
    locator = ModalResponseLocator(method=method, n_neighbors=1).fit(frame, xy)
    assert locator.selected_features_ == ["modal__m1", "modal__m2", "profile__p"]
    errors = np.hypot(*(locator.predict(frame) - xy).T)
    assert np.median(errors) == 0.0


def test_locator_feature_views_differ(rng):
    frame, xy, _ = grid_fingerprints(rng)
    assert ReceivedPowerLocator().fit(frame, xy).selected_features_ == ["power__total_dbm_mean"]
    assert RawResponseLocator().fit(frame, xy).selected_features_ == [
        "response__k0_db",
        "response__k1_db",
    ]
    assert len(RadioMapLocator().fit(frame, xy).selected_features_) == frame.shape[1]
    with pytest.raises(ValueError, match="does not contain"):
        ReceivedPowerLocator().fit(frame.drop(columns=["power__total_dbm_mean"]), xy)
    with pytest.raises(ValueError):
        RadioMapLocator(method="trilateration").fit(frame, xy)
    with pytest.raises(ValueError):
        RadioMapLocator().fit(frame, xy[:-1])


def test_benchmark_ranks_informative_features_first(rng):
    frame, xy, sessions = grid_fingerprints(rng)
    n = len(frame)
    table = FeatureTable(
        features=frame.to_numpy(),
        feature_names=list(frame.columns),
        labels=np.array(["occupied"] * n),
        recordings=np.array([f"{s}/p" for s in sessions]),
        sessions=sessions,
        blocks=np.array([f"{s}/p#b0" for s in sessions]),
        window_start_s=np.zeros(n),
        positions=xy,
    )
    benchmark = benchmark_locators(table, n_splits=3, n_bootstrap=100, seed=0)
    assert benchmark.n_folds == 3
    medians = {name: r.summary.median for name, r in benchmark.results.items()}
    assert medians["modal_response"] <= medians["raw_response"] <= medians["received_power"]
    assert benchmark.duplicates == []
    for result in benchmark.results.values():
        assert result.n_samples == n
        assert 0.0 <= result.fraction_beyond_max <= 1.0
    no_positions = FeatureTable(**{**table.__dict__, "positions": np.full((n, 2), np.nan)})
    with pytest.raises(ValueError, match="position"):
        benchmark_locators(no_positions)
