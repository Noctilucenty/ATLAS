import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer import (  # noqa: E402
    MIN_CANDLES,
    PAYOUT_MIN,
    break_even_win_rate,
    decide,
    ema,
    rsi,
)

NOW = 1_784_600_000.0


def make_candles(closes: list[float], interval: int, now: float = NOW) -> list[dict]:
    """Build completed candles ending just before `now`, oldest first."""
    n = len(closes)
    return [
        {
            "from": now - (n - i) * interval,
            "to": now - (n - i - 1) * interval,
            "close": c,
        }
        for i, c in enumerate(closes)
    ]


def trending_closes(start: float, step_up: float, step_down: float, n: int) -> list[float]:
    """Alternating +step_up / -step_down closes: a trend with pullbacks, which
    keeps Wilder RSI at a steady mid value instead of pinning to 0/100."""
    closes = [start]
    for i in range(n - 1):
        delta = step_up if i % 2 == 0 else -step_down
        closes.append(closes[-1] + delta)
    return closes


def full_inputs(closes: list[float]) -> dict:
    return {
        "1m": make_candles(closes, 60),
        "5m": make_candles(closes, 300),
        "15m": make_candles(closes, 900),
    }


UPTREND = trending_closes(1.1000, 0.0002, 0.0001, 90)   # RSI -> ~66.7
DOWNTREND = trending_closes(1.1000, 0.0001, 0.0002, 90)  # RSI -> ~33.3


# ---------------- indicators ----------------

def test_ema_of_constant_series_is_constant():
    assert ema([1.5] * 40, 8) == pytest.approx(1.5)

def test_ema_fast_tracks_recent_values_more_closely():
    rising = [float(i) for i in range(50)]
    assert ema(rising, 8) > ema(rising, 21)

def test_ema_rejects_short_series():
    with pytest.raises(ValueError):
        ema([1.0] * 5, 8)

def test_rsi_pure_uptrend_is_100():
    assert rsi([float(i) for i in range(30)]) == pytest.approx(100.0)

def test_rsi_pure_downtrend_is_0():
    assert rsi([float(30 - i) for i in range(30)]) == pytest.approx(0.0, abs=1e-9)

def test_rsi_alternating_two_up_one_down_converges_near_66():
    assert rsi(UPTREND) == pytest.approx(66.7, abs=2.0)

def test_rsi_rejects_short_series():
    with pytest.raises(ValueError):
        rsi([1.0] * 10)

def test_break_even_win_rate_at_85_percent_payout():
    assert break_even_win_rate(0.85) == pytest.approx(0.5405, abs=0.0001)


# ---------------- decide(): signals ----------------

def test_aligned_uptrend_with_good_payout_is_call():
    decision = decide(full_inputs(UPTREND), 0.85, True, NOW)
    assert decision["signal"] == "CALL"
    assert all(r.startswith(("PASS", "SKIP")) or "MOMENTUM" in r for r in decision["reasons"])
    assert 55 <= decision["metrics"]["rsi_1m"] <= 70

def test_aligned_downtrend_is_put():
    decision = decide(full_inputs(DOWNTREND), 0.85, True, NOW)
    assert decision["signal"] == "PUT"
    assert 30 <= decision["metrics"]["rsi_1m"] <= 45


# ---------------- decide(): each gate forces NO_TRADE ----------------

def test_market_closed_forces_no_trade():
    decision = decide(full_inputs(UPTREND), 0.85, False, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL MARKET_OPEN") for r in decision["reasons"])

def test_low_payout_forces_no_trade():
    decision = decide(full_inputs(UPTREND), PAYOUT_MIN - 0.05, True, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL PAYOUT") for r in decision["reasons"])

def test_missing_payout_forces_no_trade():
    decision = decide(full_inputs(UPTREND), None, True, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL PAYOUT") for r in decision["reasons"])

def test_insufficient_candles_forces_no_trade():
    inputs = full_inputs(UPTREND)
    inputs["5m"] = inputs["5m"][: MIN_CANDLES - 10]
    decision = decide(inputs, 0.85, True, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL DATA_5m") for r in decision["reasons"])

def test_stale_candles_force_no_trade():
    inputs = full_inputs(UPTREND)
    inputs["1m"] = make_candles(UPTREND, 60, now=NOW - 4000)
    decision = decide(inputs, 0.85, True, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL FRESH_1m") for r in decision["reasons"])

def test_conflicting_timeframes_force_no_trade():
    inputs = full_inputs(UPTREND)
    inputs["15m"] = make_candles(DOWNTREND, 900)
    decision = decide(inputs, 0.85, True, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL TREND_ALIGN") for r in decision["reasons"])
    assert any(r.startswith("SKIP MOMENTUM") for r in decision["reasons"])

def test_overbought_uptrend_fails_momentum_gate():
    parabolic = [1.1000 + 0.0003 * i for i in range(90)]  # all gains -> RSI 100
    decision = decide(full_inputs(parabolic), 0.85, True, NOW)
    assert decision["signal"] == "NO_TRADE"
    assert any(r.startswith("FAIL MOMENTUM") for r in decision["reasons"])

def test_empty_inputs_do_not_crash():
    decision = decide({}, None, False, NOW)
    assert decision["signal"] == "NO_TRADE"


# ---------------- determinism ----------------

def test_decide_is_deterministic():
    a = decide(full_inputs(UPTREND), 0.85, True, NOW)
    b = decide(full_inputs(UPTREND), 0.85, True, NOW)
    assert a == b
