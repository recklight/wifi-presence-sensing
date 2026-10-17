"""Reading captures: beamforming reports, amplitude tables, timelines, manifests."""

from __future__ import annotations

from .beamform_report import (
    BEAMFORM_REPORT_CODE,
    BeamformFormatError,
    BeamformReportReader,
    ReadStatistics,
    decode_antenna_permutation,
    encode_antenna_selection,
    encode_beamform_payload,
    expected_csi_length,
    pack_csi,
    parse_beamform_payload,
    read_beamform_file,
    unpack_csi,
    write_beamform_file,
)
from .dataset import DatasetManifest, ManifestError, RecordingEntry
from .rate_flags import RateInfo, decode_rate_flags, encode_rate_flags
from .snapshot import TIMESTAMP_PERIOD_US, ChannelSnapshot, RadioMeta
from .subcarriers import (
    N_SUBCARRIER_GROUPS,
    SUBCARRIER_SPACING_HZ,
    reported_subcarrier_indices,
    require_single_bandwidth,
    subcarrier_frequencies_hz,
    subcarrier_offsets_hz,
)
from .tabular_reader import (
    AmplitudeTable,
    TableSchemaError,
    contiguous_runs,
    fill_masked_entries,
    label_balance,
    load_amplitude_table,
)
from .timeline import (
    SamplingDiagnostics,
    contiguous_segments,
    find_gaps,
    sampling_diagnostics,
    unwrap_timestamps,
)
from .trace import TX_POLICIES, ChannelTrace, StackingReport, radio_setup_counts, stack_snapshots

__all__ = [
    "BEAMFORM_REPORT_CODE",
    "N_SUBCARRIER_GROUPS",
    "SUBCARRIER_SPACING_HZ",
    "TIMESTAMP_PERIOD_US",
    "TX_POLICIES",
    "AmplitudeTable",
    "BeamformFormatError",
    "BeamformReportReader",
    "ChannelSnapshot",
    "ChannelTrace",
    "DatasetManifest",
    "ManifestError",
    "RadioMeta",
    "RateInfo",
    "ReadStatistics",
    "RecordingEntry",
    "SamplingDiagnostics",
    "StackingReport",
    "TableSchemaError",
    "contiguous_runs",
    "contiguous_segments",
    "decode_antenna_permutation",
    "decode_rate_flags",
    "encode_antenna_selection",
    "encode_beamform_payload",
    "encode_rate_flags",
    "expected_csi_length",
    "fill_masked_entries",
    "find_gaps",
    "label_balance",
    "load_amplitude_table",
    "pack_csi",
    "parse_beamform_payload",
    "radio_setup_counts",
    "read_beamform_file",
    "reported_subcarrier_indices",
    "require_single_bandwidth",
    "sampling_diagnostics",
    "stack_snapshots",
    "subcarrier_frequencies_hz",
    "subcarrier_offsets_hz",
    "unpack_csi",
    "unwrap_timestamps",
    "write_beamform_file",
]
