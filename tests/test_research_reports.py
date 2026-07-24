"""Pure-function tests for the descriptive report scripts (payout landscape
and universe profiler). Descriptive infrastructure only - no labels, no WR."""

import pytest

from research_payout_landscape import break_even_wr, summarize
from research_universe_profile import series_stats


def test_break_even_wr_matches_binary_economics():
    assert break_even_wr(0.87) == pytest.approx(1 / 1.87)
    assert break_even_wr(0.92) == pytest.approx(1 / 1.92)
    # fatter payout -> lower bar to profitability
    assert break_even_wr(0.92) < break_even_wr(0.87)


def test_summarize_aggregates_per_key_and_hour():
    rows = [
        # asset, kind, ts_epoch (hour 0 and hour 1 UTC), payout
        ("EURUSD-op", "binary", 0, 0.85),
        ("EURUSD-op", "binary", 3600, 0.95),
        ("EURUSD-op", "turbo", 0, 0.50),
    ]
    out = summarize(rows)
    b = out["EURUSD-op|binary"]
    assert b["snapshots"] == 2
    assert b["mean"] == pytest.approx(0.90)
    assert b["by_utc_hour"] == {"0": 0.85, "1": 0.95}
    assert b["best_hour_utc"] == "1"
    assert out["EURUSD-op|turbo"]["snapshots"] == 1


def test_series_stats_flags_short_series():
    assert series_stats([1.0] * 5, list(range(5)))["note"] == "insufficient data"


def test_series_stats_structure_metrics():
    # alternating up/down 1% moves -> strong negative lag-1 autocorrelation
    closes, ts = [100.0], [0]
    for i in range(1, 400):
        closes.append(closes[-1] * (1.01 if i % 2 else 0.99))
        ts.append(i * 60)
    s = series_stats(closes, ts)
    assert s["candles"] == 400
    assert s["lag1_autocorr"] < -0.9
    assert s["gaps_gt_1m"] == 0
    assert s["coverage"] == pytest.approx(1.0, abs=0.01)
    # inject a 10-minute gap and it must be counted
    ts2 = ts[:200] + [t + 600 for t in ts[200:]]
    assert series_stats(closes, ts2)["gaps_gt_1m"] == 1
