"""Spatial-mapping inversion, including the complex three-stream case.

The inversion needs the conjugate transpose of the mapping matrix.  A plain
transpose gives the right answer for the real two-stream Hadamard matrix and
the wrong one for the complex three-stream DFT matrix, which is why the
inversion is checked on three streams here.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from wifi_presence_sensing.calibrate.spatial_mapping import (
    apply_spatial_mapping,
    is_unitary,
    mapping_from_phases,
    standard_mapping_matrix,
    undo_spatial_mapping,
)


def random_channel(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize(
    ("n_tx", "kind"), [(2, "hadamard"), (3, "fourier"), (4, "hadamard"), (4, "fourier")]
)
def test_round_trip_recovers_the_channel(seed: int, n_tx: int, kind: str):
    rng = np.random.default_rng(seed)
    channel = random_channel(rng, (n_tx, 3, 30))
    mapping = standard_mapping_matrix(n_tx, kind)
    recovered = undo_spatial_mapping(apply_spatial_mapping(channel, mapping), mapping)
    assert np.max(np.abs(recovered - channel)) < 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_round_trip_with_a_random_unitary_mapping(seed: int):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(random_channel(rng, (3, 3)))
    channel = random_channel(rng, (7, 3, 3, 30))  # leading packet axis is broadcast
    recovered = undo_spatial_mapping(apply_spatial_mapping(channel, q), q)
    assert np.max(np.abs(recovered - channel)) < 1e-12


def test_plain_transpose_fails_for_three_complex_streams_but_not_for_two_real():
    rng = np.random.default_rng(0)
    for n_tx, should_fail in ((2, False), (3, True)):
        mapping = standard_mapping_matrix(n_tx)
        channel = random_channel(rng, (n_tx, 3, 30))
        reported = apply_spatial_mapping(channel, mapping)
        # The mistaken recovery M Q^T (plain instead of conjugate transpose) is
        # Q C in the transmit-major layout, where conj(Q) C = Q^{-T} C is correct.
        transpose_only = np.einsum("ab,brs->ars", mapping, reported)
        error = float(np.max(np.abs(transpose_only - channel)))
        assert (error > 1e-3) == should_fail
        assert np.max(np.abs(undo_spatial_mapping(reported, mapping) - channel)) < 1e-12


def test_loop_and_vectorized_forms_agree():
    rng = np.random.default_rng(1)
    mapping = standard_mapping_matrix(3)
    reported = random_channel(rng, (3, 3, 30))
    expected = np.empty_like(reported)
    for k in range(30):
        # H_true = M Q^{-1}, with M = reported[:, :, k].T in (rx x tx) layout.
        expected[:, :, k] = (reported[:, :, k].T @ np.linalg.inv(mapping)).T
    np.testing.assert_allclose(undo_spatial_mapping(reported, mapping), expected, atol=1e-12)


def test_non_unitary_mapping_uses_the_pseudo_inverse(caplog):
    rng = np.random.default_rng(2)
    mapping = np.array([[1.0, 0.5], [0.2, 1.0]], dtype=np.complex128)
    channel = random_channel(rng, (2, 3, 30))
    with caplog.at_level(logging.WARNING):
        recovered = undo_spatial_mapping(apply_spatial_mapping(channel, mapping), mapping)
    assert "not unitary" in caplog.text
    np.testing.assert_allclose(recovered, channel, atol=1e-10)


def test_single_chain_is_returned_unchanged():
    channel = np.arange(90, dtype=np.complex128).reshape(1, 3, 30)
    out = undo_spatial_mapping(channel, np.array([[1j]]))
    np.testing.assert_array_equal(out, channel)
    assert out is not channel


def test_standard_matrices_are_unitary_and_validated():
    for n in (1, 2, 3, 4):
        assert is_unitary(standard_mapping_matrix(n))
    assert is_unitary(standard_mapping_matrix(3, "identity"))
    assert is_unitary(mapping_from_phases(np.array([[0.0, 0.0], [0.0, np.pi]])))
    with pytest.raises(ValueError):
        standard_mapping_matrix(3, "hadamard")
    with pytest.raises(ValueError):
        standard_mapping_matrix(5)
    with pytest.raises(ValueError):
        undo_spatial_mapping(np.zeros((2, 3, 30)), np.eye(3))
    with pytest.raises(ValueError):
        undo_spatial_mapping(np.zeros((3, 30)), np.eye(3))
