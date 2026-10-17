"""Mapping of the 30 reported subcarrier groups to physical OFDM subcarriers.

The Intel 5300 reports channel state for 30 subcarrier groups, using the
grouping defined for compressed beamforming feedback in IEEE Std 802.11n-2009:
every second subcarrier on a 20 MHz channel (grouping Ng = 2) and every fourth
on a 40 MHz channel (Ng = 4).  Subcarriers are spaced 312.5 kHz apart.

So "group k" on a 20 MHz capture and "group k" on a 40 MHz capture are
different physical frequencies, and traces of different bandwidths must never
be stacked along the subcarrier axis.  :func:`require_single_bandwidth`
refuses such a mixture.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: Number of subcarrier groups in every beamforming report.
N_SUBCARRIER_GROUPS = 30

#: OFDM subcarrier spacing of 802.11n, in hertz.
SUBCARRIER_SPACING_HZ = 312_500.0

_GROUPED_INDICES: dict[int, np.ndarray] = {
    # Ng = 2 on 20 MHz: -28, -26, ..., -2, -1, 1, 3, ..., 27, 28
    20: np.concatenate([np.arange(-28, -1, 2), [-1, 1], np.arange(3, 28, 2), [28]]).astype(
        np.int64
    ),
    # Ng = 4 on 40 MHz: -58, -54, ..., -2, 2, ..., 54, 58
    40: np.concatenate([np.arange(-58, 0, 4), np.arange(2, 59, 4)]).astype(np.int64),
}


def reported_subcarrier_indices(bandwidth_mhz: int) -> np.ndarray:
    """Return the OFDM subcarrier index of each reported group.

    Parameters
    ----------
    bandwidth_mhz : int
        20 or 40.

    Returns
    -------
    numpy.ndarray
        Integer array of length 30, in reporting order.

    Raises
    ------
    ValueError
        If the bandwidth is not 20 or 40 MHz.

    Examples
    --------
    >>> indices = reported_subcarrier_indices(20)
    >>> len(indices), indices[[0, 14, 15, -1]].tolist()
    (30, [-28, -1, 1, 28])
    >>> reported_subcarrier_indices(40)[[0, -1]].tolist()
    [-58, 58]
    """
    try:
        return _GROUPED_INDICES[int(bandwidth_mhz)].copy()
    except KeyError:
        raise ValueError(f"bandwidth must be 20 or 40 MHz, got {bandwidth_mhz}") from None


def subcarrier_offsets_hz(bandwidth_mhz: int) -> np.ndarray:
    """Frequency offset of each reported group from the channel centre, in hertz.

    Examples
    --------
    >>> float(subcarrier_offsets_hz(20)[-1])
    8750000.0
    """
    return reported_subcarrier_indices(bandwidth_mhz).astype(np.float64) * SUBCARRIER_SPACING_HZ


def subcarrier_frequencies_hz(bandwidth_mhz: int, center_frequency_hz: float) -> np.ndarray:
    """Absolute frequency of each reported group.

    Parameters
    ----------
    bandwidth_mhz : int
        20 or 40.
    center_frequency_hz : float
        Channel centre frequency, e.g. ``2.437e9`` for 2.4 GHz channel 6.

    Returns
    -------
    numpy.ndarray
        Frequencies in hertz, length 30.
    """
    if center_frequency_hz <= 0:
        raise ValueError(f"center frequency must be positive, got {center_frequency_hz}")
    return center_frequency_hz + subcarrier_offsets_hz(bandwidth_mhz)


def require_single_bandwidth(bandwidths: Iterable[int]) -> int:
    """Return the common bandwidth of a collection, refusing a mixture.

    Parameters
    ----------
    bandwidths : iterable of int
        Bandwidth of each trace or recording.

    Returns
    -------
    int
        The single bandwidth shared by all inputs.

    Raises
    ------
    ValueError
        If the collection is empty or mixes bandwidths.

    Examples
    --------
    >>> require_single_bandwidth([20, 20, 20])
    20
    """
    values = sorted({int(b) for b in bandwidths})
    if not values:
        raise ValueError("no bandwidth to check: the collection is empty")
    if len(values) > 1:
        raise ValueError(
            f"captures mix channel bandwidths {values} MHz; subcarrier group k maps to a "
            "different physical frequency in each, so they cannot share a feature axis. "
            "Analyse each bandwidth separately (set capture.bandwidth_mhz in the config)."
        )
    return values[0]
