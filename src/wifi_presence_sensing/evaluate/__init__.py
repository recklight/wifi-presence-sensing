"""Evaluation: grouped splits, classification and localization metrics, error CDFs."""

from __future__ import annotations

from .crossval import CrossValidationReport, compare_classifiers, cross_validate_grouped
from .error_cdf import (
    ErrorCdfResult,
    error_cdf_from_errors,
    error_distance_cdf,
    find_duplicate_curves,
)
from .metrics import (
    ClassificationMetrics,
    ErrorSummary,
    classification_metrics,
    localization_errors,
    sensitivity_specificity,
    summarize_errors,
)
from .splitting import (
    SplitError,
    assert_disjoint_groups,
    grouped_folds,
    grouped_stratified_folds,
    groups_per_class,
)

__all__ = [
    "ClassificationMetrics",
    "CrossValidationReport",
    "ErrorCdfResult",
    "ErrorSummary",
    "SplitError",
    "assert_disjoint_groups",
    "classification_metrics",
    "compare_classifiers",
    "cross_validate_grouped",
    "error_cdf_from_errors",
    "error_distance_cdf",
    "find_duplicate_curves",
    "grouped_folds",
    "grouped_stratified_folds",
    "groups_per_class",
    "localization_errors",
    "sensitivity_specificity",
    "summarize_errors",
]
