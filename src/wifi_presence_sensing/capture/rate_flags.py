"""Decoding of the 16-bit rate-and-flags word carried by each beamforming report.

The Intel 5300 firmware describes the PHY rate of the frame that triggered the
report with a bit field (``rate_n_flags`` in the driver's terms).  The layout
used here is:

========  ======  ===========================================================
bits      mask    meaning
========  ======  ===========================================================
0-6       0x007F  HT: MCS index (0-7 one stream, 8-15 two, 16-23 three, 32 dup)
                  non-HT: the PLCP rate code (bits 0-7)
8         0x0100  HT (802.11n) frame
9         0x0200  CCK (802.11b) modulation, only defined when bit 8 is clear
10        0x0400  greenfield preamble
11        0x0800  40 MHz (HT40) channel
12        0x1000  duplicate data on both 20 MHz halves (requires HT40)
13        0x2000  short guard interval (0.4 us instead of 0.8 us)
========  ======  ===========================================================

For HT frames the low byte is an MCS index (MCS 15 is ``0x0F``), not a rate
code, and the CCK bit (bit 9) is only defined when the HT bit is clear;
:func:`decode_rate_flags` therefore reads the HT bit first and interprets the
rest of the word accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

HT_FLAG = 0x0100
CCK_FLAG = 0x0200
GREENFIELD_FLAG = 0x0400
HT40_FLAG = 0x0800
DUPLICATE_FLAG = 0x1000
SHORT_GUARD_INTERVAL_FLAG = 0x2000
MCS_MASK = 0x007F
NON_HT_RATE_MASK = 0x00FF

#: MCS 32 is the single-stream HT40 duplicate format.
DUPLICATE_MCS = 32


@dataclass(frozen=True)
class RateInfo:
    """Decoded view of the rate-and-flags word.

    Attributes
    ----------
    raw : int
        The original 16-bit value.
    high_throughput : bool
        True for 802.11n (HT) frames.
    mcs : int or None
        HT modulation-and-coding-scheme index, ``None`` for non-HT frames.
    bandwidth_mhz : int
        20 or 40.
    short_guard_interval : bool
        True when the 0.4 us guard interval was used.
    greenfield : bool
        True for a greenfield preamble.
    duplicate : bool
        True when data was duplicated on both 20 MHz halves of an HT40 channel.
    cck : bool
        True for an 802.11b CCK frame.  Always False for HT frames.
    non_ht_rate_code : int or None
        The PLCP rate code of a non-HT frame, ``None`` for HT frames.
    """

    raw: int
    high_throughput: bool
    mcs: int | None
    bandwidth_mhz: int
    short_guard_interval: bool
    greenfield: bool
    duplicate: bool
    cck: bool
    non_ht_rate_code: int | None

    @property
    def n_spatial_streams(self) -> int | None:
        """Number of spatial streams implied by the MCS index (HT frames only)."""
        if self.mcs is None:
            return None
        if self.mcs == DUPLICATE_MCS:
            return 1
        if 0 <= self.mcs <= 31:
            return self.mcs // 8 + 1
        return None


def decode_rate_flags(rate_field: int) -> RateInfo:
    """Decode the rate-and-flags word of a beamforming report.

    Parameters
    ----------
    rate_field : int
        The 16-bit word as stored in the report.

    Returns
    -------
    RateInfo
        The decoded flags.

    Raises
    ------
    ValueError
        If the value does not fit in 16 bits.

    Examples
    --------
    MCS 15 (two streams) on a 20 MHz channel; in an HT frame the low byte is
    the MCS index, so the CCK flag stays False:

    >>> info = decode_rate_flags(0x010F)
    >>> info.high_throughput, info.mcs, info.n_spatial_streams, info.cck
    (True, 15, 2, False)
    >>> decode_rate_flags(0x0100 | 0x0800 | 0x2000 | 12).bandwidth_mhz
    40
    """
    value = int(rate_field)
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"the rate-and-flags word must fit in 16 bits, got {value:#x}")

    high_throughput = bool(value & HT_FLAG)
    ht40 = bool(value & HT40_FLAG)
    return RateInfo(
        raw=value,
        high_throughput=high_throughput,
        mcs=(value & MCS_MASK) if high_throughput else None,
        bandwidth_mhz=40 if ht40 else 20,
        short_guard_interval=bool(value & SHORT_GUARD_INTERVAL_FLAG),
        greenfield=bool(value & GREENFIELD_FLAG),
        duplicate=bool(value & DUPLICATE_FLAG),
        # The CCK bit is only meaningful for non-HT frames; for HT frames the
        # low byte is an MCS index and bit 9 is reserved.
        cck=(not high_throughput) and bool(value & CCK_FLAG),
        non_ht_rate_code=None if high_throughput else (value & NON_HT_RATE_MASK),
    )


def encode_rate_flags(
    mcs: int,
    *,
    bandwidth_mhz: int = 20,
    short_guard_interval: bool = False,
    greenfield: bool = False,
) -> int:
    """Build an HT rate-and-flags word; the inverse of :func:`decode_rate_flags`.

    Parameters
    ----------
    mcs : int
        HT MCS index, 0-32.
    bandwidth_mhz : int, optional
        20 or 40.
    short_guard_interval, greenfield : bool, optional
        Flags to set.

    Returns
    -------
    int
        The encoded 16-bit value.

    Raises
    ------
    ValueError
        If the MCS index or the bandwidth is out of range.

    Examples
    --------
    >>> hex(encode_rate_flags(14))
    '0x10e'
    >>> decode_rate_flags(encode_rate_flags(11, bandwidth_mhz=40)).bandwidth_mhz
    40
    """
    if not 0 <= mcs <= DUPLICATE_MCS:
        raise ValueError(f"HT MCS index must be in 0..32, got {mcs}")
    if bandwidth_mhz not in (20, 40):
        raise ValueError(f"bandwidth must be 20 or 40 MHz, got {bandwidth_mhz}")
    value = HT_FLAG | mcs
    if bandwidth_mhz == 40:
        value |= HT40_FLAG
    if short_guard_interval:
        value |= SHORT_GUARD_INTERVAL_FLAG
    if greenfield:
        value |= GREENFIELD_FLAG
    return value
