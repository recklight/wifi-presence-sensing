from __future__ import annotations

import pytest

from wifi_presence_sensing.capture.rate_flags import (
    CCK_FLAG,
    HT40_FLAG,
    HT_FLAG,
    SHORT_GUARD_INTERVAL_FLAG,
    decode_rate_flags,
    encode_rate_flags,
)


@pytest.mark.parametrize("mcs", range(16))
def test_ht_low_byte_is_the_mcs_index_and_never_cck(mcs: int):
    info = decode_rate_flags(HT_FLAG | mcs)
    assert info.high_throughput
    assert info.mcs == mcs
    assert info.cck is False
    assert info.non_ht_rate_code is None


def test_reserved_bit_nine_is_ignored_for_ht_frames():
    info = decode_rate_flags(HT_FLAG | CCK_FLAG | 15)
    assert info.cck is False


def test_cck_is_decoded_only_for_non_ht_frames():
    info = decode_rate_flags(CCK_FLAG | 0x0A)
    assert not info.high_throughput
    assert info.cck is True
    assert info.mcs is None
    assert info.non_ht_rate_code == 0x0A


@pytest.mark.parametrize(
    ("mcs", "streams"), [(0, 1), (7, 1), (8, 2), (14, 2), (15, 2), (16, 3), (32, 1)]
)
def test_spatial_streams_follow_from_the_mcs_index(mcs: int, streams: int):
    assert decode_rate_flags(HT_FLAG | mcs).n_spatial_streams == streams


def test_bandwidth_and_guard_interval_flags():
    info = decode_rate_flags(HT_FLAG | HT40_FLAG | SHORT_GUARD_INTERVAL_FLAG | 11)
    assert info.bandwidth_mhz == 40
    assert info.short_guard_interval
    assert decode_rate_flags(HT_FLAG | 11).bandwidth_mhz == 20


@pytest.mark.parametrize("bandwidth", [20, 40])
@pytest.mark.parametrize("mcs", [0, 9, 15])
def test_encode_and_decode_are_inverse(mcs: int, bandwidth: int):
    raw = encode_rate_flags(mcs, bandwidth_mhz=bandwidth, short_guard_interval=True)
    info = decode_rate_flags(raw)
    assert (info.mcs, info.bandwidth_mhz, info.short_guard_interval) == (mcs, bandwidth, True)


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        decode_rate_flags(0x1_0000)
    with pytest.raises(ValueError):
        encode_rate_flags(40)
    with pytest.raises(ValueError):
        encode_rate_flags(3, bandwidth_mhz=80)
