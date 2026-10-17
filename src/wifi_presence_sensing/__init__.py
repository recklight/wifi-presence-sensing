"""WiFi CSI analysis for indoor occupancy sensing and fingerprint localization.

The pipeline reads 802.11n beamforming reports, undoes the transmitter's
spatial mapping, puts the CSI on an absolute scale and resamples it onto
uniform grids.  Each antenna pair's amplitude series is then decomposed with
(ensemble) empirical mode decomposition, and the intrinsic mode functions
become features for occupancy classification and fingerprint localization.
Both tasks are evaluated with grouped cross-validation against baselines.

Subpackages
-----------
capture
    Capture files, amplitude tables, timelines, dataset manifests.
calibrate
    Spatial mapping, absolute scaling, noise model, link quality, resampling.
decompose
    EMD, EEMD and CEEMDAN with pluggable stopping criteria.
features
    Modal statistics, subcarrier profiles and windowed feature tables.
occupancy
    Occupancy classifier and its baselines.
localize
    Radio-map fingerprint locators and their benchmark.
evaluate
    Grouped splits, classification and localization metrics, error CDFs.
report
    Figures with companion tables and run manifests.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .calibrate import (
    MODULATION_MODELS,
    NoiseModel,
    amplitude_db,
    apply_spatial_mapping,
    bpsk_error_rate,
    bpsk_error_rate_inverse,
    detrended_phase,
    effective_snr,
    qam16_error_rate,
    qam16_error_rate_inverse,
    qam64_error_rate,
    qam64_error_rate_inverse,
    qpsk_error_rate,
    qpsk_error_rate_inverse,
    resample_uniform,
    standard_mapping_matrix,
    to_calibrated_csi,
    total_received_power_dbm,
    undo_spatial_mapping,
)
from .capture import (
    BeamformReportReader,
    ChannelSnapshot,
    ChannelTrace,
    DatasetManifest,
    RadioMeta,
    decode_rate_flags,
    load_amplitude_table,
    read_beamform_file,
    sampling_diagnostics,
    stack_snapshots,
    unwrap_timestamps,
    write_beamform_file,
)
from .config import Config, ConfigError
from .decompose import (
    CauchyStopping,
    EnsembleDecomposition,
    FixedIterationStopping,
    ModeDecomposition,
    ThreeThresholdStopping,
    decompose,
    decompose_ceemdan,
    decompose_eemd,
    decompose_emd,
)
from .evaluate import (
    classification_metrics,
    compare_classifiers,
    cross_validate_grouped,
    error_distance_cdf,
    grouped_stratified_folds,
)
from .features import (
    AmplitudeSeries,
    FeatureTable,
    build_feature_table,
    modal_feature_vector,
    reduce_subcarriers,
)
from .localize import (
    ModalResponseLocator,
    RadioMapLocator,
    RawResponseLocator,
    ReceivedPowerLocator,
    benchmark_locators,
)
from .occupancy import OccupancyClassifier, candidate_classifiers

__all__ = [
    "MODULATION_MODELS",
    "AmplitudeSeries",
    "BeamformReportReader",
    "CauchyStopping",
    "ChannelSnapshot",
    "ChannelTrace",
    "Config",
    "ConfigError",
    "DatasetManifest",
    "EnsembleDecomposition",
    "FeatureTable",
    "FixedIterationStopping",
    "ModalResponseLocator",
    "ModeDecomposition",
    "NoiseModel",
    "OccupancyClassifier",
    "RadioMapLocator",
    "RadioMeta",
    "RawResponseLocator",
    "ReceivedPowerLocator",
    "ThreeThresholdStopping",
    "__version__",
    "amplitude_db",
    "apply_spatial_mapping",
    "benchmark_locators",
    "bpsk_error_rate",
    "bpsk_error_rate_inverse",
    "build_feature_table",
    "candidate_classifiers",
    "classification_metrics",
    "compare_classifiers",
    "cross_validate_grouped",
    "decode_rate_flags",
    "decompose",
    "decompose_ceemdan",
    "decompose_eemd",
    "decompose_emd",
    "detrended_phase",
    "effective_snr",
    "error_distance_cdf",
    "grouped_stratified_folds",
    "load_amplitude_table",
    "modal_feature_vector",
    "qam16_error_rate",
    "qam16_error_rate_inverse",
    "qam64_error_rate",
    "qam64_error_rate_inverse",
    "qpsk_error_rate",
    "qpsk_error_rate_inverse",
    "read_beamform_file",
    "reduce_subcarriers",
    "resample_uniform",
    "sampling_diagnostics",
    "stack_snapshots",
    "standard_mapping_matrix",
    "to_calibrated_csi",
    "total_received_power_dbm",
    "undo_spatial_mapping",
    "unwrap_timestamps",
    "write_beamform_file",
]
