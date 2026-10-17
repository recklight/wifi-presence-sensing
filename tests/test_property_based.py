"""Property-based checks (run only when hypothesis is installed: pip install .[test]).

The same properties are covered by seed sweeps in the other test files, so
nothing essential depends on this module.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from wifi_presence_sensing.calibrate.link_quality import MODULATION_MODELS  # noqa: E402
from wifi_presence_sensing.calibrate.spatial_mapping import (  # noqa: E402
    apply_spatial_mapping,
    standard_mapping_matrix,
    undo_spatial_mapping,
)
from wifi_presence_sensing.capture.beamform_report import pack_csi, unpack_csi  # noqa: E402


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n_tx=st.sampled_from([2, 3, 4]), n_rx=st.integers(1, 3))
def test_spatial_mapping_round_trip(seed: int, n_tx: int, n_rx: int):
    rng = np.random.default_rng(seed)
    channel = rng.normal(size=(n_tx, n_rx, 30)) + 1j * rng.normal(size=(n_tx, n_rx, 30))
    mapping = standard_mapping_matrix(n_tx)
    recovered = undo_spatial_mapping(apply_spatial_mapping(channel, mapping), mapping)
    assert np.max(np.abs(recovered - channel)) < 1e-12


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n_tx=st.integers(1, 3),
    n_rx=st.integers(1, 3),
)
def test_bit_packing_round_trip(seed: int, n_tx: int, n_rx: int):
    rng = np.random.default_rng(seed)
    csi = rng.integers(-128, 128, (n_tx, n_rx, 30)) + 1j * rng.integers(-128, 128, (n_tx, n_rx, 30))
    np.testing.assert_array_equal(unpack_csi(pack_csi(csi), n_tx, n_rx), csi)


@settings(max_examples=100, deadline=None)
@given(
    snr=st.floats(1e-3, 1e4, allow_nan=False),
    modulation=st.sampled_from(sorted(MODULATION_MODELS)),
)
def test_error_rate_round_trip_in_the_log_domain(snr: float, modulation: str):
    model = MODULATION_MODELS[modulation]
    recovered = model.snr_from_log_error_rate(model.log_error_rate(snr))
    assert recovered == pytest.approx(snr, rel=1e-8)
