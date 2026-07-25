"""The feature cache must be impossible to poison.

A research pipeline silently reading stale features would be far worse than
the minutes the cache saves, so the key hashes the candle VALUES plus every
output-affecting parameter. These tests pin that: any change to data or
parameters must change the key, and a cached frame must equal a fresh build.
"""

import pandas as pd
import pytest

from feature_cache import cached_build, signature


def _candles(n: int = 700, base: float = 1.1000, step: float = 0.00001):
    ts = [1_784_000_000 + 60 * i for i in range(n)]
    close = [base + step * ((i % 17) - 8) for i in range(n)]
    return pd.DataFrame({
        "from_ts": ts,
        "to_ts": [t + 60 for t in ts],
        "open": close,
        "high": [c + 0.00005 for c in close],
        "low": [c - 0.00005 for c in close],
        "close": close,
        "volume": [10.0] * n,
    })


def test_identical_input_gives_identical_key():
    a, b = _candles(), _candles()
    assert signature(a, "EURUSD", 60, 15) == signature(b, "EURUSD", 60, 15)


def test_one_changed_candle_changes_the_key():
    """The decisive property: not row count, not latest timestamp - VALUES."""
    a = _candles()
    b = a.copy()
    b.loc[300, "close"] = b.loc[300, "close"] + 0.00001
    assert len(a) == len(b)
    assert a["to_ts"].max() == b["to_ts"].max()
    assert signature(a, "EURUSD", 60, 15) != signature(b, "EURUSD", 60, 15)


def test_appended_data_changes_the_key():
    a = _candles(700)
    b = _candles(701)
    assert signature(a, "EURUSD", 60, 15) != signature(b, "EURUSD", 60, 15)


@pytest.mark.parametrize("kwargs", [
    {"asset": "GBPUSD"}, {"interval": 300}, {"horizon": 5},
])
def test_parameters_change_the_key(kwargs):
    c = _candles()
    base = dict(asset="EURUSD", interval=60, horizon=15)
    assert signature(c, **base) != signature(c, **{**base, **kwargs})


def test_flags_change_the_key():
    c = _candles()
    k1 = signature(c, "EURUSD", 60, 15, entry_next_open=True, extra_vol=True)
    k2 = signature(c, "EURUSD", 60, 15, entry_next_open=True, extra_vol=False)
    k3 = signature(c, "EURUSD", 60, 15, entry_next_open=False, extra_vol=True)
    assert len({k1, k2, k3}) == 3


def test_flag_order_does_not_change_the_key():
    c = _candles()
    a = signature(c, "EURUSD", 60, 15, extra_vol=True, entry_next_open=True)
    b = signature(c, "EURUSD", 60, 15, entry_next_open=True, extra_vol=True)
    assert a == b


def test_cached_frame_equals_a_fresh_build(tmp_path, monkeypatch):
    import feature_cache
    from features import build_features

    monkeypatch.setattr(feature_cache, "CACHE_DIR", tmp_path / "f")
    c = _candles()
    fresh = build_features(c, interval=60, horizon=15, entry_next_open=True,
                           extra_vol=True)
    first = cached_build(c, "EURUSD", 60, 15, entry_next_open=True,
                         extra_vol=True)          # miss -> builds and writes
    second = cached_build(c, "EURUSD", 60, 15, entry_next_open=True,
                          extra_vol=True)         # hit -> reads parquet
    pd.testing.assert_frame_equal(fresh, first)
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True),
                                  second.reset_index(drop=True))


def test_no_cache_bypasses_storage_entirely(tmp_path, monkeypatch):
    import feature_cache

    cache_dir = tmp_path / "f"
    monkeypatch.setattr(feature_cache, "CACHE_DIR", cache_dir)
    cached_build(_candles(), "EURUSD", 60, 15, use_cache=False,
                 entry_next_open=True, extra_vol=True)
    assert not cache_dir.exists() or not list(cache_dir.glob("*.parquet"))


def test_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    import feature_cache

    monkeypatch.setattr(feature_cache, "CACHE_DIR", tmp_path / "f")
    c = _candles()
    cached_build(c, "EURUSD", 60, 15, entry_next_open=True, extra_vol=True)
    files = list((tmp_path / "f").glob("*.parquet"))
    assert files
    files[0].write_bytes(b"not a parquet file")
    out = cached_build(c, "EURUSD", 60, 15, entry_next_open=True,
                       extra_vol=True)
    assert not out.empty      # rebuilt rather than raising
