"""Deterministic EURUSD binary-option signal analyzer.

Pure functions only - no network, no clock, no randomness. Every decision is a
function of the inputs, and every gate emits a PASS/FAIL reason string so a
reviewer can reproduce the exact verdict from the journaled inputs.

The strategy never modifies itself: all constants are frozen per
STRATEGY_VERSION and may only change through a reviewed version bump.
"""

STRATEGY_VERSION = "1.0.0"

ASSET = "EURUSD"
TF_INTERVALS = {"1m": 60, "5m": 300, "15m": 900}  # label -> candle size in seconds

PAYOUT_MIN = 0.80        # break-even win rate at 0.80 payout = 55.6%
EMA_FAST = 8
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_CALL_BAND = (55.0, 70.0)  # momentum up, but not overbought
RSI_PUT_BAND = (30.0, 45.0)   # momentum down, but not oversold
MIN_CANDLES = 60         # completed candles required per timeframe
FRESHNESS_FACTOR = 2     # last completed candle must be <= factor*interval old


def ema(values: list[float], period: int) -> float:
    """Exponential moving average (SMA-seeded), returning the latest value."""
    if len(values) < period:
        raise ValueError(f"need >= {period} values, got {len(values)}")
    k = 2.0 / (period + 1)
    current = sum(values[:period]) / period
    for v in values[period:]:
        current = v * k + current * (1 - k)
    return current


def rsi(values: list[float], period: int = RSI_PERIOD) -> float:
    """Wilder-smoothed RSI, returning the latest value."""
    if len(values) < period + 1:
        raise ValueError(f"need >= {period + 1} values, got {len(values)}")
    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def break_even_win_rate(payout: float) -> float:
    """Win rate needed for zero expected value at a given payout ratio."""
    return 1.0 / (1.0 + payout)


def decide(candles_by_tf: dict, payout, market_open: bool, now: float) -> dict:
    """Evaluate all gates and return {'signal', 'reasons', 'metrics'}.

    signal is 'CALL', 'PUT' or 'NO_TRADE'. Any single gate failure forces
    NO_TRADE. candles_by_tf maps '1m'/'5m'/'15m' to raw candle dicts with at
    least 'to' and 'close' keys; `now` is the epoch timestamp the candles were
    fetched at (passed in, never read from a clock, for reproducibility).
    """
    reasons: list[str] = []
    metrics: dict = {"strategy_version": STRATEGY_VERSION, "payout": payout}
    failed = False

    def gate(name: str, ok: bool, detail: str) -> None:
        nonlocal failed
        reasons.append(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failed = True

    gate("MARKET_OPEN", bool(market_open), f"{ASSET} turbo market open={bool(market_open)}")

    if payout is None:
        gate("PAYOUT", False, "payout unavailable")
    else:
        be = break_even_win_rate(payout)
        metrics["break_even_win_rate"] = round(be, 4)
        gate(
            "PAYOUT",
            payout >= PAYOUT_MIN,
            f"payout={payout:.2f} (min {PAYOUT_MIN:.2f}); break-even win rate={be:.1%}",
        )

    trends: dict[str, str] = {}
    closes_1m: list[float] = []
    for label, interval in TF_INTERVALS.items():
        candles = candles_by_tf.get(label) or []
        completed = [c for c in candles if c["to"] <= now]
        closes = [float(c["close"]) for c in completed]
        gate(
            f"DATA_{label}",
            len(closes) >= MIN_CANDLES,
            f"{len(closes)} completed candles (need {MIN_CANDLES})",
        )
        if len(closes) < MIN_CANDLES:
            continue
        age = now - completed[-1]["to"]
        gate(
            f"FRESH_{label}",
            age <= FRESHNESS_FACTOR * interval,
            f"last completed candle closed {age:.0f}s ago (max {FRESHNESS_FACTOR * interval}s)",
        )
        fast = ema(closes, EMA_FAST)
        slow = ema(closes, EMA_SLOW)
        trend = "up" if fast > slow else "down" if fast < slow else "flat"
        trends[label] = trend
        metrics[label] = {
            "ema_fast": fast,
            "ema_slow": slow,
            "trend": trend,
            "completed_candles": len(closes),
        }
        if label == "1m":
            closes_1m = closes

    aligned = (
        len(trends) == len(TF_INTERVALS)
        and len(set(trends.values())) == 1
        and "flat" not in trends.values()
    )
    gate("TREND_ALIGN", aligned, f"trends={trends or 'insufficient data'}")

    direction = None
    if aligned:
        direction = "CALL" if trends["1m"] == "up" else "PUT"
        rsi_1m = rsi(closes_1m, RSI_PERIOD)
        metrics["rsi_1m"] = round(rsi_1m, 2)
        band = RSI_CALL_BAND if direction == "CALL" else RSI_PUT_BAND
        gate(
            "MOMENTUM",
            band[0] <= rsi_1m <= band[1],
            f"1m RSI({RSI_PERIOD})={rsi_1m:.1f}, required band for {direction}={band}",
        )
    else:
        reasons.append("SKIP MOMENTUM: no aligned trend to confirm")

    signal = "NO_TRADE" if failed else direction
    return {"signal": signal, "reasons": reasons, "metrics": metrics}
