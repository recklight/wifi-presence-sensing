"""Feature extraction: modal statistics, subcarrier profiles, windowed feature tables."""

from __future__ import annotations

from .extraction import (
    GROUP_SEPARATOR,
    META_COLUMNS,
    OPTIONAL_META_COLUMNS,
    AmplitudeSeries,
    FeatureTable,
    build_feature_table,
    stream_names,
    window_features,
    window_starts,
)
from .modal_statistics import (
    MODAL_STATISTICS,
    instantaneous_frequency,
    modal_feature_vector,
    mode_statistics,
)
from .subcarrier_profile import (
    PROFILE_METRICS,
    REDUCTION_METHODS,
    profile_feature_vector,
    reduce_subcarriers,
    subcarrier_profile,
)

__all__ = [
    "GROUP_SEPARATOR",
    "META_COLUMNS",
    "MODAL_STATISTICS",
    "OPTIONAL_META_COLUMNS",
    "PROFILE_METRICS",
    "REDUCTION_METHODS",
    "AmplitudeSeries",
    "FeatureTable",
    "build_feature_table",
    "instantaneous_frequency",
    "modal_feature_vector",
    "mode_statistics",
    "profile_feature_vector",
    "reduce_subcarriers",
    "stream_names",
    "subcarrier_profile",
    "window_features",
    "window_starts",
]
