"""Pipeline settings as frozen dataclasses, one per configuration section.

The capture policy, calibration, resampling grid, decomposition, feature
windows, classifier, cross-validation layout and output folder are all set
here rather than in the code that uses them.  A configuration loads from YAML
and takes dotted-key overrides from ``--set``; unknown keys and invalid values
are rejected with a message naming the key.  Every run folder gets a copy as
``config.yaml``, so a result can be reproduced from its own folder.

Examples
--------
>>> config = Config()
>>> config.decomposition.method
'eemd'
>>> config.override({"decomposition.n_ensemble": 8}).decomposition.n_ensemble
8
"""

from __future__ import annotations

import dataclasses
import logging
import numbers
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from .calibrate.noise_model import NoiseStrategy
from .calibrate.resample import RESAMPLE_METHODS
from .calibrate.spatial_mapping import MAPPING_KINDS
from .capture.tabular_reader import TABLE_LAYOUTS
from .capture.trace import TX_POLICIES
from .decompose import DECOMPOSITION_METHODS
from .decompose.stopping import (
    STOPPING_CRITERIA,
    StoppingCriterion,
    stopping_from_name,
)
from .features.subcarrier_profile import REDUCTION_METHODS
from .localize.fingerprint import LOCATOR_METHODS
from .occupancy.classifier import OCCUPANCY_MODELS

_LOGGER = logging.getLogger(__name__)

#: Feature groups produced by the feature extraction.
FEATURE_GROUPS: tuple[str, ...] = ("power", "response", "profile", "modal")

#: Ways of forming cross-validation groups.
GROUPINGS: tuple[str, ...] = ("recording", "session", "block")

#: Spatial mapping settings: the standard kinds, "none", or "phases" (custom).
SPATIAL_MAPPINGS: tuple[str, ...] = ("none", *MAPPING_KINDS, "phases")

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or out of range."""


def _choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _number(name: str, value: object) -> None:
    # A bool from YAML or --set would compare as 1 or 0 and pass as a number.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{name} must be a number, got {value!r}")


def _numbers(name: str, values: object) -> tuple[float, ...]:
    # Lists from YAML arrive as tuples (see _tuplify); a list passed in code is fine too.
    if not isinstance(values, (tuple, list)):
        raise ConfigError(f"{name} must be a list of numbers, got {values!r}")
    for value in values:
        _number(name, value)
    return tuple(values)


def _increasing_pair(name: str, values: object) -> None:
    pair = _numbers(name, values)
    if len(pair) != 2 or not pair[0] < pair[1]:
        raise ConfigError(f"{name} must be an increasing [low, high] pair, got {list(pair)}")


def _positive(name: str, value: float) -> None:
    _number(name, value)
    if not value > 0:
        raise ConfigError(f"{name} must be strictly positive, got {value!r}")


def _integer(name: str, value: object, minimum: int) -> None:
    # bool is a subclass of int, and "--set x=true" must not pass as a count of 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}")


def _flag(name: str, value: object) -> None:
    # A string such as "no" would be truthy and silently mean the opposite.
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class CaptureConfig:
    """How beamforming reports become a trace.

    Attributes
    ----------
    tx_policy : str
        ``dominant`` keeps packets with the most common number of transmit
        chains; ``first_chain`` keeps transmit chain 0 of every packet.
    apply_antenna_permutation : bool
        Reorder receive columns into physical antenna order.
    bandwidth_mhz : int or None
        Analyse only captures at this bandwidth.  ``None`` requires every
        capture of a dataset to share one bandwidth.
    """

    tx_policy: str = "dominant"
    apply_antenna_permutation: bool = True
    bandwidth_mhz: int | None = None

    def __post_init__(self) -> None:
        _choice("capture.tx_policy", self.tx_policy, TX_POLICIES)
        _flag("capture.apply_antenna_permutation", self.apply_antenna_permutation)
        if self.bandwidth_mhz not in (None, 20, 40):
            raise ConfigError(
                f"capture.bandwidth_mhz must be 20, 40 or null, got {self.bandwidth_mhz}"
            )


@dataclass(frozen=True)
class CalibrationConfig:
    """Spatial-mapping inversion, absolute scaling and the noise model.

    Attributes
    ----------
    spatial_mapping : str
        ``none`` (direct mapping), ``identity``, ``hadamard``, ``fourier``,
        or ``phases`` to use :attr:`mapping_phases_rad`.
    mapping_phases_rad : tuple of tuple of float or None
        Square phase table for a custom constant-modulus mapping.
    noise_strategy : str
        ``reported``, ``sentinel_guard`` or ``robust``.
    noise_sentinel_dbm, noise_fallback_dbm : float
        Sentinel meaning "not measured" and the floor substituted for it.
    noise_threshold : float
        Hampel threshold in scaled MADs.
    noise_window : int or None
        Rolling Hampel window (odd, at least 3), ``None`` for trace-wide
        statistics.
    noise_plausible_range_dbm : (float, float)
        Final clamp of the noise floor.
    include_quantization_noise : bool
        Count the 8-bit rounding noise in SNR calibration.
    """

    spatial_mapping: str = "none"
    mapping_phases_rad: tuple[tuple[float, ...], ...] | None = None
    noise_strategy: str = "robust"
    noise_sentinel_dbm: float = -127.0
    noise_fallback_dbm: float = -92.0
    noise_threshold: float = 3.5
    noise_window: int | None = None
    noise_plausible_range_dbm: tuple[float, float] = (-105.0, -60.0)
    include_quantization_noise: bool = True

    def __post_init__(self) -> None:
        _choice("calibration.spatial_mapping", self.spatial_mapping, SPATIAL_MAPPINGS)
        _choice(
            "calibration.noise_strategy",
            self.noise_strategy,
            tuple(s.value for s in NoiseStrategy),
        )
        if self.spatial_mapping == "phases" and not self.mapping_phases_rad:
            raise ConfigError("calibration.spatial_mapping='phases' needs mapping_phases_rad")
        if self.mapping_phases_rad is not None:
            rows = self.mapping_phases_rad
            if not isinstance(rows, (tuple, list)) or not all(
                len(_numbers("calibration.mapping_phases_rad", row)) == len(rows) for row in rows
            ):
                raise ConfigError(
                    "calibration.mapping_phases_rad must be a square table of phases in radians, "
                    f"got {rows!r}"
                )
        _number("calibration.noise_sentinel_dbm", self.noise_sentinel_dbm)
        _number("calibration.noise_fallback_dbm", self.noise_fallback_dbm)
        _positive("calibration.noise_threshold", self.noise_threshold)
        if self.noise_window is not None:
            _integer("calibration.noise_window", self.noise_window, 3)
            if self.noise_window % 2 == 0:
                raise ConfigError(f"calibration.noise_window must be odd, got {self.noise_window}")
        _flag("calibration.include_quantization_noise", self.include_quantization_noise)
        _increasing_pair("calibration.noise_plausible_range_dbm", self.noise_plausible_range_dbm)


@dataclass(frozen=True)
class ResampleConfig:
    """Uniform resampling grid.

    Attributes
    ----------
    rate_hz : float
        Target rate.  Pick it no higher than the long-run packet rate.
    max_gap_s : float
        Longer pauses split the trace into segments.
    method : str
        ``auto``, ``linear`` or ``bin_mean``.
    min_segment_s : float
        Shorter segments are discarded.
    """

    rate_hz: float = 10.0
    max_gap_s: float = 1.0
    method: str = "auto"
    min_segment_s: float = 10.0

    def __post_init__(self) -> None:
        _positive("resample.rate_hz", self.rate_hz)
        _positive("resample.max_gap_s", self.max_gap_s)
        _choice("resample.method", self.method, RESAMPLE_METHODS)
        _number("resample.min_segment_s", self.min_segment_s)
        if self.min_segment_s < 0:
            raise ConfigError("resample.min_segment_s must be non-negative")


@dataclass(frozen=True)
class DecompositionConfig:
    """Empirical mode decomposition settings.

    Attributes
    ----------
    method : str
        ``emd``, ``eemd`` or ``ceemdan``.
    n_ensemble : int
        Noise realizations (ensemble methods).
    noise_scale : float
        Added noise standard deviation relative to the signal's.
    max_modes : int
        IMFs per series; also the number of modes in the feature layout.
    stopping : str
        ``three_threshold``, ``cauchy`` or ``fixed``.
    theta1, theta2, alpha : float
        Three-threshold parameters.
    cauchy_tolerance : float
        Cauchy criterion tolerance.
    fixed_iterations : int
        Sifting steps for the fixed criterion.
    max_sift_iterations : int
        Hard cap on sifting steps per IMF.
    paired_noise : bool
        EEMD with each noise realization used at both signs.
    """

    method: str = "eemd"
    n_ensemble: int = 50
    noise_scale: float = 0.2
    max_modes: int = 4
    stopping: str = "three_threshold"
    theta1: float = 0.05
    theta2: float = 0.5
    alpha: float = 0.05
    cauchy_tolerance: float = 0.2
    fixed_iterations: int = 10
    max_sift_iterations: int = 1000
    paired_noise: bool = False

    def __post_init__(self) -> None:
        _choice("decomposition.method", self.method, DECOMPOSITION_METHODS)
        _choice("decomposition.stopping", self.stopping, STOPPING_CRITERIA)
        _integer("decomposition.n_ensemble", self.n_ensemble, 1)
        _number("decomposition.noise_scale", self.noise_scale)
        if self.noise_scale < 0:
            raise ConfigError("decomposition.noise_scale must be non-negative")
        _integer("decomposition.max_modes", self.max_modes, 1)
        _integer("decomposition.fixed_iterations", self.fixed_iterations, 1)
        _integer("decomposition.max_sift_iterations", self.max_sift_iterations, 1)
        _flag("decomposition.paired_noise", self.paired_noise)
        for key in ("theta1", "theta2", "alpha", "cauchy_tolerance"):
            _number(f"decomposition.{key}", getattr(self, key))
        # Building the criterion validates its parameters.
        try:
            self.stopping_criterion()
        except ValueError as exc:
            raise ConfigError(f"decomposition stopping parameters: {exc}") from None

    def stopping_criterion(self) -> StoppingCriterion:
        """Instantiate the configured stopping criterion."""
        if self.stopping == "three_threshold":
            return stopping_from_name(
                "three_threshold", theta1=self.theta1, theta2=self.theta2, alpha=self.alpha
            )
        if self.stopping == "cauchy":
            return stopping_from_name("cauchy", tolerance=self.cauchy_tolerance)
        return stopping_from_name("fixed", n_iterations=self.fixed_iterations)


@dataclass(frozen=True)
class FeatureConfig:
    """Windowing and feature extraction.

    Attributes
    ----------
    window_s : float
        Window length.
    hop_s : float
        Step between window starts.
    stream_reduction : str
        How the subcarriers of an antenna pair become one series to
        decompose: ``pca`` or ``mean``.
    block_s : float
        Length of the time blocks used by ``group_by: block``.  Windows that
        straddle a block boundary are dropped so blocks never share samples.
    n_jobs : int
        Parallel workers over windows (``-1`` for all cores).
    """

    window_s: float = 12.8
    hop_s: float = 6.4
    stream_reduction: str = "pca"
    block_s: float = 60.0
    n_jobs: int = 1

    def __post_init__(self) -> None:
        _positive("features.window_s", self.window_s)
        _positive("features.hop_s", self.hop_s)
        _positive("features.block_s", self.block_s)
        _choice("features.stream_reduction", self.stream_reduction, REDUCTION_METHODS)
        if self.block_s < self.window_s:
            raise ConfigError("features.block_s must be at least one window long")
        # joblib counts negative values back from the number of cores; 0 means no worker.
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ConfigError(
                "features.n_jobs must be a non-zero integer (1: no parallelism, -1: every "
                f"core), got {self.n_jobs!r}"
            )


@dataclass(frozen=True)
class TableConfig:
    """Schema of an amplitude table exported as CSV.

    Attributes
    ----------
    path : str
        CSV file (placeholder; point it at your own export).
    n_label_columns : int
        Leading integer label columns.
    label_column : int
        Which label column is the class label.
    n_streams, n_subcarriers : int
        Shape of the amplitude block.
    layout : str
        ``stream_major`` or ``subcarrier_major``.
    sentinel_values : tuple of float
        Values meaning "missing".
    valid_range_db : (float, float)
        Amplitudes outside are masked.
    nominal_rate_hz : float
        Row rate of the table, used as the sampling rate of each run.
    """

    path: str = "data/raw/amplitude_table.csv"
    n_label_columns: int = 2
    label_column: int = 0
    n_streams: int = 6
    n_subcarriers: int = 30
    layout: str = "stream_major"
    sentinel_values: tuple[float, ...] = (65535.0,)
    valid_range_db: tuple[float, float] = (-40.0, 80.0)
    nominal_rate_hz: float = 1.0

    def __post_init__(self) -> None:
        _choice("table.layout", self.layout, TABLE_LAYOUTS)
        _integer("table.n_label_columns", self.n_label_columns, 1)
        _integer("table.label_column", self.label_column, 0)
        _integer("table.n_streams", self.n_streams, 1)
        _integer("table.n_subcarriers", self.n_subcarriers, 1)
        if not 0 <= self.label_column < self.n_label_columns:
            raise ConfigError("table.label_column must index one of the label columns")
        _positive("table.nominal_rate_hz", self.nominal_rate_hz)
        _numbers("table.sentinel_values", self.sentinel_values)
        _increasing_pair("table.valid_range_db", self.valid_range_db)


@dataclass(frozen=True)
class OccupancyConfig:
    """The occupancy classifier.

    Attributes
    ----------
    model : str
        ``logistic``, ``random_forest``, ``svm``, ``mlp`` or ``majority``.
    feature_groups : tuple of str
        Feature groups the classifier sees.
    regularization : float
        Inverse regularization strength C (logistic, svm); the MLP uses an L2
        penalty of ``1e-4 / regularization``.
    n_estimators : int
        Trees of the random forest.
    positive_label : str or None
        Label treated as "positive" for sensitivity, specificity and ROC-AUC.
    """

    model: str = "logistic"
    feature_groups: tuple[str, ...] = ("modal", "profile")
    regularization: float = 1.0
    n_estimators: int = 300
    positive_label: str | None = "occupied"

    def __post_init__(self) -> None:
        _choice("occupancy.model", self.model, OCCUPANCY_MODELS)
        for group in self.feature_groups:
            _choice("occupancy.feature_groups", group, FEATURE_GROUPS)
        _positive("occupancy.regularization", self.regularization)
        _integer("occupancy.n_estimators", self.n_estimators, 1)


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation layout.

    Attributes
    ----------
    n_splits : int
        Folds of the stratified group k-fold.
    group_by : str
        ``recording``, ``session`` or ``block``.
    nested_search : bool
        Tune the classifier's regularization in an inner grouped CV.
    inner_splits : int
        Folds of the inner loop.
    """

    n_splits: int = 5
    group_by: str = "session"
    nested_search: bool = False
    inner_splits: int = 3

    def __post_init__(self) -> None:
        _integer("evaluation.n_splits", self.n_splits, 2)
        _integer("evaluation.inner_splits", self.inner_splits, 2)
        _flag("evaluation.nested_search", self.nested_search)
        _choice("evaluation.group_by", self.group_by, GROUPINGS)


@dataclass(frozen=True)
class LocalizationConfig:
    """Fingerprint localization benchmark.

    Attributes
    ----------
    method : str
        ``knn`` (deterministic radio-map matching) or ``probabilistic``.
    n_neighbors : int
        Reference points averaged into each estimate.
    cdf_thresholds_m : tuple of float
        Error thresholds at which the CDF is tabulated, strictly increasing.
    n_bootstrap : int
        Bootstrap resamples for the CDF confidence band.
    confidence : float
        Confidence level of the band.
    group_by : str
        Cross-validation groups.
    """

    method: str = "knn"
    n_neighbors: int = 3
    cdf_thresholds_m: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    n_bootstrap: int = 1000
    confidence: float = 0.95
    group_by: str = "session"

    def __post_init__(self) -> None:
        _choice("localization.method", self.method, LOCATOR_METHODS)
        _choice("localization.group_by", self.group_by, GROUPINGS)
        _integer("localization.n_neighbors", self.n_neighbors, 1)
        thresholds = _numbers("localization.cdf_thresholds_m", self.cdf_thresholds_m)
        if not thresholds or any(t <= 0 for t in thresholds):
            raise ConfigError("localization.cdf_thresholds_m must be positive distances")
        if any(b <= a for a, b in pairwise(thresholds)):
            raise ConfigError("localization.cdf_thresholds_m must be strictly increasing")
        _number("localization.confidence", self.confidence)
        if not 0 < self.confidence < 1:
            raise ConfigError("localization.confidence must lie in (0, 1)")
        _integer("localization.n_bootstrap", self.n_bootstrap, 0)


@dataclass(frozen=True)
class OutputConfig:
    """Where results go.

    Attributes
    ----------
    root : str
        Output folder; every run writes into a time-stamped subfolder.
    figure_format : str
        ``png``, ``pdf`` or ``svg``.
    """

    root: str = "outputs"
    figure_format: str = "png"

    def __post_init__(self) -> None:
        _choice("output.figure_format", self.figure_format, ("png", "pdf", "svg"))


_SECTIONS: dict[str, type] = {
    "capture": CaptureConfig,
    "calibration": CalibrationConfig,
    "resample": ResampleConfig,
    "decomposition": DecompositionConfig,
    "features": FeatureConfig,
    "table": TableConfig,
    "occupancy": OccupancyConfig,
    "evaluation": EvaluationConfig,
    "localization": LocalizationConfig,
    "output": OutputConfig,
}


def _tuplify(value: Any) -> Any:
    """Lists from YAML become tuples so the frozen dataclasses stay hashable."""
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _listify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_listify(v) for v in value]
    return value


def _build_section(name: str, raw: Any) -> Any:
    cls = _SECTIONS[name]
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"section {name!r} must be a mapping")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {name!r}: {sorted(unknown)}; known: {sorted(known)}")
    try:
        return cls(**{key: _tuplify(value) for key, value in raw.items()})
    except TypeError as exc:
        raise ConfigError(f"section {name!r}: {exc}") from None


@dataclass(frozen=True)
class Config:
    """Complete configuration of a run.

    Attributes
    ----------
    seed : int
        Master seed: decompositions, folds and bootstrap all derive from it.
    log_level : str
        Logging threshold.
    dataset : str
        Dataset manifest (YAML) used by ``prepare``; a placeholder until you
        point it at your own captures.
    data_root : str or None
        Base folder for relative paths in the manifest (defaults to the
        manifest's folder).
    """

    seed: int = 42
    log_level: str = "INFO"
    dataset: str = "configs/datasets/example_occupancy.yaml"
    data_root: str | None = None
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    resample: ResampleConfig = field(default_factory=ResampleConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    table: TableConfig = field(default_factory=TableConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if str(self.log_level).upper() not in _LOG_LEVELS:
            raise ConfigError(f"log_level={self.log_level!r} is not a logging level")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {self.seed!r}")
        mapping = self.calibration.spatial_mapping
        if self.capture.tx_policy == "first_chain" and mapping not in ("none", "identity"):
            # Row 0 of a mapped report is a mixture of every transmit chain, and
            # the mapping can only be undone on the full (n_tx, n_rx) tensor.
            raise ConfigError(
                "capture.tx_policy='first_chain' keeps transmit chain 0 of the reported CSI, "
                f"which still mixes all chains under calibration.spatial_mapping={mapping!r}; "
                "use capture.tx_policy='dominant' to undo a spatial mapping"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        """Build a configuration from a nested mapping, rejecting unknown keys."""
        data = dict(raw or {})
        top_level = {"seed", "log_level", "dataset", "data_root"}
        unknown = set(data) - top_level - set(_SECTIONS)
        if unknown:
            raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")
        kwargs: dict[str, Any] = {key: data[key] for key in top_level if key in data}
        for name in _SECTIONS:
            kwargs[name] = _build_section(name, data.get(name))
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a configuration file."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"configuration file not found: {source}")
        with source.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested mapping (lists instead of tuples), ready for YAML or JSON."""
        return _listify_tree(dataclasses.asdict(self))

    def save(self, path: str | Path) -> Path:
        """Write the configuration as YAML."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return destination

    def override(self, updates: dict[str, Any]) -> Config:
        """Return a copy with dotted keys replaced, e.g. ``{"resample.rate_hz": 5}``."""
        tree = self.to_dict()
        for dotted, value in updates.items():
            parts = dotted.split(".")
            node = tree
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise ConfigError(f"unknown configuration section in {dotted!r}")
                node = node[part]
            if parts[-1] not in node:
                raise ConfigError(f"unknown configuration key {dotted!r}")
            node[parts[-1]] = value
        return Config.from_dict(tree)


def _listify_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _listify_tree(v) for k, v in value.items()}
    return _listify(value)
