"""Inverting the 802.11n transmit spatial mapping.

An 802.11n transmitter may multiply its space-time streams by a spatial
mapping matrix :math:`Q` before they reach the antennas (IEEE Std 802.11n-2009,
"spatial mapping").  The receiver estimates the channel from training fields
that went through the same matrix, so the reported channel is

.. math::

    M = H\\,Q, \\qquad H, M \\in \\mathbb{C}^{N_{rx} \\times N_{tx}},

and the physical channel is recovered as :math:`H = M Q^{-1}`.  For a unitary
:math:`Q`, :math:`Q^{-1} = Q^{H}`, the conjugate transpose.

CSI tensors here are stored transmit-major, ``(n_tx, n_rx, n_subcarriers)``,
i.e. each subcarrier holds :math:`C = M^{T}`.  Transposing the recovery gives

.. math::

    C_{true} = (M Q^{-1})^{T} = Q^{-T} C = \\overline{Q}\\, C
    \\quad (Q \\text{ unitary}),

applied to all subcarriers at once with a single :func:`numpy.einsum`.

The conjugate only matters when :math:`Q` is complex.  The two-stream Hadamard
matrix is real, so the plain transpose happens to give the same recovery
there.  The three-stream DFT matrix contains :math:`e^{\\pm 2\\pi i/3}`, and
on it the plain transpose returns a scrambled channel;
``tests/test_spatial_mapping.py`` covers that case.

A matrix that is not unitary (for instance a measured, vendor-specific one)
is inverted with the Moore-Penrose pseudo-inverse and a warning is logged.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

_LOGGER = logging.getLogger(__name__)

#: Spatial mapping kinds understood by :func:`standard_mapping_matrix`.
MAPPING_KINDS: tuple[str, ...] = ("identity", "hadamard", "fourier")


@lru_cache(maxsize=32)
def _cached_mapping(n_streams: int, kind: str) -> np.ndarray:
    if kind == "identity":
        matrix = np.eye(n_streams, dtype=np.complex128)
    elif kind == "hadamard":
        if n_streams & (n_streams - 1):
            raise ValueError(
                f"a Hadamard mapping needs a power-of-two stream count, got {n_streams}; "
                "use kind='fourier' for three streams"
            )
        matrix = np.ones((1, 1))
        while matrix.shape[0] < n_streams:
            matrix = np.block([[matrix, matrix], [matrix, -matrix]])
        matrix = matrix.astype(np.complex128) / np.sqrt(n_streams)
    elif kind == "fourier":
        k = np.arange(n_streams)
        matrix = np.exp(-2j * np.pi * np.outer(k, k) / n_streams) / np.sqrt(n_streams)
    else:
        raise ValueError(f"mapping kind {kind!r} is not one of {MAPPING_KINDS}")
    matrix.setflags(write=False)
    return matrix


def standard_mapping_matrix(n_streams: int, kind: str | None = None) -> np.ndarray:
    """Return a unitary spatial mapping matrix of the kinds described in 802.11n.

    Parameters
    ----------
    n_streams : int
        Matrix size (1, 2, 3 or 4).
    kind : {"identity", "hadamard", "fourier"}, optional
        ``identity`` is direct mapping.  ``hadamard`` and ``fourier`` are the
        orthonormal Hadamard and DFT matrices.  The default is ``hadamard``
        for power-of-two sizes and ``fourier`` otherwise.

    Returns
    -------
    numpy.ndarray
        Read-only complex matrix shaped ``(n_streams, n_streams)``.

    Raises
    ------
    ValueError
        If the size or the kind is not supported.

    Examples
    --------
    >>> q = standard_mapping_matrix(2)
    >>> np.allclose(q * np.sqrt(2), [[1, 1], [1, -1]])
    True
    >>> q3 = standard_mapping_matrix(3)
    >>> np.allclose(q3.conj().T @ q3, np.eye(3)), np.allclose(q3.T @ q3, np.eye(3))
    (True, False)
    """
    if not 1 <= int(n_streams) <= 4:
        raise ValueError(f"802.11n supports 1 to 4 streams, got {n_streams}")
    n = int(n_streams)
    if kind is None:
        kind = "hadamard" if n & (n - 1) == 0 else "fourier"
    return _cached_mapping(n, kind)


def mapping_from_phases(phases_rad: np.ndarray) -> np.ndarray:
    """Build a constant-modulus mapping ``exp(1j * phases) / sqrt(n)``.

    Hardware often uses its own unitary mapping; this constructor lets such a
    matrix be written in a configuration file as a table of phases.

    Parameters
    ----------
    phases_rad : numpy.ndarray
        Square array of phases in radians.

    Returns
    -------
    numpy.ndarray
        Complex mapping matrix.

    Examples
    --------
    >>> m = mapping_from_phases(np.array([[0.0, 0.0], [0.0, np.pi]]))
    >>> np.allclose(m, standard_mapping_matrix(2, "hadamard"))
    True
    """
    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.ndim != 2 or phases.shape[0] != phases.shape[1]:
        raise ValueError(f"phases must form a square matrix, got shape {phases.shape}")
    return np.exp(1j * phases) / np.sqrt(phases.shape[0])


def is_unitary(matrix: np.ndarray, *, atol: float = 1e-10) -> bool:
    """True when ``matrix^H @ matrix`` is the identity within ``atol``.

    Examples
    --------
    >>> is_unitary(standard_mapping_matrix(3)), is_unitary(np.array([[1.0, 1.0], [0.0, 1.0]]))
    (True, False)
    """
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return bool(np.allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=atol))


def _recovery_operator(mapping: np.ndarray) -> np.ndarray:
    """Return :math:`Q^{-T}`, the matrix that left-multiplies a transmit-major tensor."""
    if is_unitary(mapping):
        return mapping.conj()
    _LOGGER.warning(
        "spatial mapping matrix is not unitary; inverting it with the pseudo-inverse "
        "(condition number %.3g)",
        np.linalg.cond(mapping),
    )
    return np.linalg.pinv(mapping).T


def _check_tensor(csi: np.ndarray, mapping: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = np.asarray(csi)
    matrix = np.asarray(mapping, dtype=np.complex128)
    if tensor.ndim < 3:
        raise ValueError(
            f"CSI must be shaped (..., n_tx, n_rx, n_subcarriers); got {tensor.ndim} dimension(s)"
        )
    n_tx = tensor.shape[-3]
    if matrix.shape != (n_tx, n_tx):
        raise ValueError(
            f"mapping must be {n_tx}x{n_tx} to match n_tx={n_tx}, got shape {matrix.shape}"
        )
    return tensor, matrix


def undo_spatial_mapping(csi: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Remove the transmitter's spatial mapping from reported CSI.

    Parameters
    ----------
    csi : numpy.ndarray
        Complex CSI shaped ``(..., n_tx, n_rx, n_subcarriers)``; any leading
        axes (e.g. packets) are broadcast.
    mapping : numpy.ndarray
        The ``n_tx x n_tx`` spatial mapping matrix :math:`Q` used by the
        transmitter.

    Returns
    -------
    numpy.ndarray
        The physical channel, same shape as ``csi``.  A single transmit chain
        is returned unchanged (a 1x1 mapping is a pure phase, absorbed by the
        channel).

    Raises
    ------
    ValueError
        If the shapes do not agree.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> h = rng.normal(size=(3, 3, 30)) + 1j * rng.normal(size=(3, 3, 30))
    >>> q = standard_mapping_matrix(3)
    >>> np.allclose(undo_spatial_mapping(apply_spatial_mapping(h, q), q), h)
    True
    """
    tensor, matrix = _check_tensor(csi, mapping)
    if tensor.shape[-3] == 1:
        return np.array(tensor, dtype=np.complex128, copy=True)
    recovery = _recovery_operator(matrix)
    return np.einsum("ab,...brs->...ars", recovery, tensor)


def apply_spatial_mapping(csi: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Apply a spatial mapping to a physical channel: :math:`C_{rep} = Q^{T} C_{true}`.

    The forward model of :func:`undo_spatial_mapping`, used to simulate
    reports and to test the inversion.

    Parameters
    ----------
    csi : numpy.ndarray
        Physical channel shaped ``(..., n_tx, n_rx, n_subcarriers)``.
    mapping : numpy.ndarray
        The ``n_tx x n_tx`` mapping matrix.

    Returns
    -------
    numpy.ndarray
        The channel as the receiver would report it.
    """
    tensor, matrix = _check_tensor(csi, mapping)
    if tensor.shape[-3] == 1:
        return np.array(tensor, dtype=np.complex128, copy=True)
    return np.einsum("ab,...brs->...ars", matrix.T, tensor)
