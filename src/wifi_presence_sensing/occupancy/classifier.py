"""Occupancy classifier and the baselines it has to beat.

:class:`OccupancyClassifier` is a scikit-learn estimator: it picks feature
groups out of a :class:`~wifi_presence_sensing.features.FeatureTable` frame,
standardizes them and fits one of several interchangeable models.  It follows
the estimator protocol (``get_params``/``clone``/``fit``/``predict``), so
grouped cross-validation and grid search take it as is.

Whether the modal features help can only be judged against baselines, so
:func:`candidate_classifiers` always evaluates the proposed model next to

``majority``
    Always predicts the most frequent training class (chance level).
``mean_amplitude``
    Logistic regression on the single feature ``response__mean_db``.  The
    average CSI amplitude barely differs between an empty and an occupied
    room, which is why the temporal-variation features are there; this
    baseline shows how far the average alone gets on a given dataset.
``raw_response``
    The same model on the time-averaged amplitude of every subcarrier.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features.extraction import GROUP_SEPARATOR

_LOGGER = logging.getLogger(__name__)

#: Models accepted by :class:`OccupancyClassifier`.
OCCUPANCY_MODELS: tuple[str, ...] = ("logistic", "random_forest", "svm", "mlp", "majority")


class OccupancyClassifier(ClassifierMixin, BaseEstimator):
    """Standardize selected feature groups and classify occupancy.

    Parameters
    ----------
    model : str
        ``logistic``, ``random_forest``, ``svm``, ``mlp`` or ``majority``.
    feature_groups : tuple of str or None
        Feature groups to use when ``X`` is a DataFrame (``None``: all).
    feature_columns : tuple of str or None
        Exact column names; overrides ``feature_groups``.
    regularization : float
        Inverse regularization strength ``C`` (logistic, svm); the MLP uses an
        L2 penalty of ``1e-4 / regularization``.
    n_estimators : int
        Trees of the random forest.
    seed : int
        Random state of every stochastic model.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame({"modal__a": np.r_[rng.normal(0, 1, 50), rng.normal(3, 1, 50)],
    ...                   "response__mean_db": rng.normal(0, 1, 100)})
    >>> y = np.array(["empty"] * 50 + ["occupied"] * 50)
    >>> model = OccupancyClassifier(feature_groups=("modal",)).fit(X, y)
    >>> model.selected_features_, bool(model.score(X, y) > 0.9)
    (['modal__a'], True)
    """

    def __init__(
        self,
        model: str = "logistic",
        feature_groups: tuple[str, ...] | None = ("modal", "profile"),
        feature_columns: tuple[str, ...] | None = None,
        regularization: float = 1.0,
        n_estimators: int = 300,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.feature_groups = feature_groups
        self.feature_columns = feature_columns
        self.regularization = regularization
        self.n_estimators = n_estimators
        self.seed = seed

    def _build(self, labels: np.ndarray) -> Pipeline:
        if self.model not in OCCUPANCY_MODELS:
            raise ValueError(f"model={self.model!r} is not one of {OCCUPANCY_MODELS}")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        estimator: Any
        if self.model == "logistic":
            estimator = LogisticRegression(
                C=self.regularization, max_iter=5000, class_weight="balanced"
            )
        elif self.model == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=self.n_estimators, class_weight="balanced", random_state=self.seed
            )
        elif self.model == "svm":
            # Platt scaling through CalibratedClassifierCV: SVC(probability=True)
            # is deprecated from scikit-learn 1.9.  ensemble=False fits one SVC on
            # all training data and calibrates it on out-of-fold scores.  Five
            # folds need five windows per class, which a small table or the
            # inner folds of a nested search may not have, so the fold count
            # drops to what the smallest class allows (at least 2).
            smallest_class = int(np.unique(labels, return_counts=True)[1].min())
            estimator = CalibratedClassifierCV(
                SVC(C=self.regularization, kernel="rbf", class_weight="balanced"),
                cv=max(2, min(5, smallest_class)),
                ensemble=False,
            )
        elif self.model == "mlp":
            estimator = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-4 / self.regularization,
                max_iter=2000,
                early_stopping=False,
                random_state=self.seed,
            )
        else:
            estimator = DummyClassifier(strategy="most_frequent")
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    def _select(self, X: pd.DataFrame | np.ndarray, *, fitting: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fitting:
                if self.feature_columns:
                    missing = [c for c in self.feature_columns if c not in X.columns]
                    if missing:
                        raise ValueError(f"feature column(s) {missing} not in the table")
                    columns = list(self.feature_columns)
                elif self.feature_groups:
                    prefixes = tuple(f"{g}{GROUP_SEPARATOR}" for g in self.feature_groups)
                    columns = [c for c in X.columns if str(c).startswith(prefixes)]
                else:
                    columns = [str(c) for c in X.columns]
                if not columns:
                    available = sorted({str(c).split(GROUP_SEPARATOR)[0] for c in X.columns})
                    raise ValueError(
                        f"no column matches feature groups {self.feature_groups}; "
                        f"available prefixes: {available}"
                    )
                self.selected_features_ = columns
            columns = self.selected_features_
            if columns is None:
                data = X.to_numpy(dtype=np.float64)
            else:
                missing = [c for c in columns if c not in X.columns]
                if missing:
                    raise ValueError(f"input lacks the trained feature column(s) {missing[:5]}")
                data = X[columns].to_numpy(dtype=np.float64)
        else:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim != 2:
                raise ValueError("X must be two-dimensional")
            if fitting:
                self.selected_features_ = None
        if not np.all(np.isfinite(data)):
            raise ValueError("features contain NaN or infinite values")
        return data

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[object]) -> OccupancyClassifier:
        """Fit on a feature frame (or array) and labels."""
        data = self._select(X, fitting=True)
        labels = np.asarray([str(v) for v in y])
        if labels.size != data.shape[0]:
            raise ValueError("X and y have different numbers of samples")
        if np.unique(labels).size < 2 and self.model != "majority":
            raise ValueError("training data must contain at least two classes")
        self.pipeline_ = self._build(labels).fit(data, labels)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicted labels."""
        return self.pipeline_.predict(self._select(X, fitting=False))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Class probabilities in :attr:`classes_` order."""
        return self.pipeline_.predict_proba(self._select(X, fitting=False))


def baseline_classifiers(seed: int = 0) -> dict[str, OccupancyClassifier]:
    """The reference models every proposed classifier is compared with."""
    return {
        "majority": OccupancyClassifier(model="majority", feature_groups=None, seed=seed),
        "mean_amplitude": OccupancyClassifier(
            model="logistic", feature_columns=("response__mean_db",), seed=seed
        ),
    }


def candidate_classifiers(
    *,
    model: str = "logistic",
    feature_groups: Sequence[str] = ("modal", "profile"),
    regularization: float = 1.0,
    n_estimators: int = 300,
    seed: int = 0,
) -> dict[str, OccupancyClassifier]:
    """Proposed classifier, the raw-amplitude variant and the baselines."""
    common = {
        "model": model,
        "regularization": regularization,
        "n_estimators": n_estimators,
        "seed": seed,
    }
    return {
        "modal": OccupancyClassifier(feature_groups=tuple(feature_groups), **common),
        "raw_response": OccupancyClassifier(feature_groups=("response",), **common),
        **baseline_classifiers(seed),
    }


def save_classifier(
    classifier: OccupancyClassifier, path: str | Path, metadata: dict[str, Any] | None = None
) -> Path:
    """Persist a fitted classifier with joblib, together with free-form metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "metadata": metadata or {}}, destination)
    return destination


def load_classifier(path: str | Path) -> tuple[OccupancyClassifier, dict[str, Any]]:
    """Load a classifier written by :func:`save_classifier`.

    Only load files you created yourself: joblib files can execute code.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model file not found: {source}")
    payload = joblib.load(source)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(f"{source} is not a saved occupancy classifier")
    return payload["classifier"], payload.get("metadata", {})
