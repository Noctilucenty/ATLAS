"""One-shot EURUSD analysis + optional single $1 PRACTICE trade.

Flow: connect -> force PRACTICE balance (hard abort if not) -> market-open
check -> payout check -> fetch 1m/5m/15m candles -> analyzer.decide() ->
if CALL/PUT, place one $1 one-minute binary and wait for the result ->
journal everything. NO_TRADE is a normal, journaled outcome.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

from analyzer import ASSET, STRATEGY_VERSION, TF_INTERVALS, decide
from journal import open_journal, record_run

PROJECT_DIR = Path(__file__).resolve().parent
TRADE_AMOUNT = 1.0
TRADE_DURATION_MIN = 1
CANDLE_COUNT = 120


def _call(fn, *args, timeout=60, **kwargs):
    """Run a blocking library call on a daemon thread with a hard timeout.

    The iqoptionapi library busy-waits forever on lost replies; a daemon
    thread lets the process exit even if the call never returns."""
    box = {}
    done = threading.Event()

    def runner():
        try:
            box["value"] = fn(*args, **kwargs)
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=runner, daemon=True).start()
    if not done.wait(timeout):
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} timed out after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _load_env() -> None:
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env()
    from iqoptionapi.stable_api import IQ_Option

    client = IQ_Option(os.environ["IQ_EMAIL"], os.environ["IQ_PASSWORD"])
    ok, reason = _call(client.connect, timeout=90)
    if not ok:
        print(json.dumps({"error": f"login failed: {reason}"}))
        return 1

    # Hard practice-only guard, independent of any env flag.
    _call(client.change_balance, "PRACTICE")
    mode = _call(client.get_balance_mode)
    if mode != "PRACTICE":
        print(json.dumps({"error": f"refusing to run: balance mode is {mode}"}))
        return 1

    # Broker naming: spot forex binaries are keyed '<PAIR>-op' in the payout
    # and open-time tables (see collector.payout_candidates).
    from collector import payout_candidates

    market_open_error = payout_error = None
    market_open = False
    try:
        open_time = _call(client.get_all_open_time, timeout=90)
        for key in payout_candidates(ASSET):
            entry = open_time["turbo"].get(key) or {}
            if entry.get("open"):
                market_open = True
                break
    except Exception as exc:
        market_open_error = f"{type(exc).__name__}: {exc}"

    payout = None
    try:
        profits = _call(client.get_all_profit, timeout=90)
        for key in payout_candidates(ASSET):
            value = profits.get(key, {}).get("turbo") if key in profits else None
            if isinstance(value, (int, float)):
                payout = float(value)
                break
        if payout is None:
            payout_error = f"no turbo payout under any of {payout_candidates(ASSET)}"
    except Exception as exc:
        payout_error = f"{type(exc).__name__}: {exc}"

    now = time.time()
    candles = {}
    for label, interval in TF_INTERVALS.items():
        candles[label] = _call(client.get_candles, ASSET, interval, CANDLE_COUNT, now)

    decision = decide(candles, payout, market_open, now)
    if market_open_error:
        decision["metrics"]["market_open_error"] = market_open_error
    if payout_error:
        decision["metrics"]["payout_error"] = payout_error

    order_id = result = profit = None
    if decision["signal"] in ("CALL", "PUT"):
        ok, order_id = _call(
            client.buy,
            TRADE_AMOUNT,
            ASSET,
            decision["signal"].lower(),
            TRADE_DURATION_MIN,
            timeout=60,
        )
        if ok:
            result, profit = _call(
                client.check_win_v4, order_id, timeout=TRADE_DURATION_MIN * 60 + 60
            )
        else:
            decision["reasons"].append(f"FAIL ORDER: broker rejected the trade ({order_id})")
            order_id = None

    balance_after = _call(client.get_balance)

    conn = open_journal()
    run_id = record_run(
        conn,
        strategy_version=STRATEGY_VERSION,
        asset=ASSET,
        signal=decision["signal"],
        reasons=decision["reasons"],
        metrics=decision["metrics"],
        payout=payout,
        market_open=market_open,
        balance_mode=mode,
        candles=candles,
        order_id=order_id,
        amount=TRADE_AMOUNT if order_id else None,
        duration_minutes=TRADE_DURATION_MIN if order_id else None,
        result=result,
        profit=profit,
        balance_after=balance_after,
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "signal": decision["signal"],
                "reasons": decision["reasons"],
                "metrics": decision["metrics"],
                "market_open": market_open,
                "payout": payout,
                "balance_mode": mode,
                "order_id": order_id,
                "result": result,
                "profit": profit,
                "balance_after": balance_after,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
