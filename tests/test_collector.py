import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector import (  # noqa: E402
    dedupe_candles,
    find_gaps,
    normalize_candle,
    plan_pages,
)


def candle(from_ts: int, interval: int = 60, close: float = 1.1) -> dict:
    return {
        "from_ts": from_ts,
        "to_ts": from_ts + interval,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0,
    }


# ---------------- plan_pages ----------------

def test_single_page_when_range_fits():
    pages = plan_pages(end_ts=1_000_000, hours=1, interval=60, page_size=1000)
    assert pages == [1_000_000]

def test_multiple_pages_step_backwards_by_page_size():
    pages = plan_pages(end_ts=1_000_000, hours=50, interval=60, page_size=1000)
    assert pages[0] == 1_000_000
    assert pages[1] == 1_000_000 - 1000 * 60
    assert len(pages) == 4  # 3001 candles -> 4 pages

def test_pages_cover_requested_span():
    hours, interval = 100, 300
    pages = plan_pages(end_ts=2_000_000, hours=hours, interval=interval, page_size=1000)
    oldest_reachable = pages[-1] - 1000 * interval
    assert oldest_reachable <= 2_000_000 - hours * 3600


# ---------------- normalize_candle ----------------

def test_normalize_maps_broker_fields():
    raw = {"from": 100.0, "to": 160.0, "open": 1.0, "max": 1.2, "min": 0.9, "close": 1.1, "volume": 5}
    normalized = normalize_candle(raw)
    assert normalized == {
        "from_ts": 100, "to_ts": 160, "open": 1.0, "high": 1.2,
        "low": 0.9, "close": 1.1, "volume": 5.0,
    }

def test_normalize_tolerates_missing_volume():
    raw = {"from": 100, "to": 160, "open": 1, "max": 1, "min": 1, "close": 1, "volume": None}
    assert normalize_candle(raw)["volume"] == 0.0


# ---------------- dedupe ----------------

def test_dedupe_removes_duplicates_and_sorts():
    candles = [candle(300), candle(100), candle(300), candle(200)]
    deduped = dedupe_candles(candles)
    assert [c["from_ts"] for c in deduped] == [100, 200, 300]

def test_dedupe_keeps_first_occurrence():
    a = candle(100, close=1.5)
    b = candle(100, close=9.9)
    assert dedupe_candles([a, b])[0]["close"] == 1.5


# ---------------- gaps ----------------

def test_contiguous_series_has_no_gaps():
    series = [candle(t) for t in range(0, 600, 60)]
    assert find_gaps(series, 60) == []

def test_single_missing_candle_detected():
    series = [candle(0), candle(60), candle(180)]  # 120 missing
    gaps = find_gaps(series, 60)
    assert gaps == [{"after_ts": 60, "resume_ts": 180, "missing": 1}]

def test_multi_candle_gap_counts_all_missing():
    series = [candle(0), candle(300)]  # 60,120,180,240 missing
    gaps = find_gaps(series, 60)
    assert gaps[0]["missing"] == 4

def test_empty_and_singleton_series():
    assert find_gaps([], 60) == []
    assert find_gaps([candle(0)], 60) == []
