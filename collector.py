"""Historical candle and payout collector.

Collects paginated candle history and prospective payout snapshots into
DuckDB (market.duckdb, see storage.py). Datasets are immutable: every
collection run creates a new tagged dataset row (asset, timeframe, collection
time, source) and its candles are never updated afterwards. Every dataset
must pass Pandera validation (validation.py) before it is stored. Payouts
cannot be reconstructed historically, so snapshots are taken prospectively -
any simulation over periods without snapshots must state its assumed payout
explicitly.

Pure helpers (plan_pages / dedupe_candles / find_gaps / normalize_candle)
have no network or clock dependencies and are unit-tested.

Usage:
  python collector.py candles EURUSD --interval 60 --hours 48
  python collector.py payouts
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

PAGE_SIZE = 1000  # broker maximum per get_candles request


# ---------------- pure helpers ----------------

def plan_pages(end_ts: float, hours: float, interval: int, page_size: int = PAGE_SIZE) -> list[float]:
    """End timestamps for each backwards page request, newest page first."""
    total = int(hours * 3600 // interval) + 1
    pages = []
    cursor = end_ts
    remaining = total
    while remaining > 0:
        pages.append(cursor)
        step = min(page_size, remaining)
        cursor -= step * interval
        remaining -= step
    return pages


def normalize_candle(raw: dict) -> dict:
    """Map a raw broker candle to the canonical UTC-epoch schema."""
    return {
        "from_ts": int(raw["from"]),
        "to_ts": int(raw["to"]),
        "open": float(raw["open"]),
        "high": float(raw["max"]),
        "low": float(raw["min"]),
        "close": float(raw["close"]),
        "volume": float(raw.get("volume") or 0),
    }


def dedupe_candles(candles: list[dict]) -> list[dict]:
    """Drop duplicate from_ts entries (first occurrence wins), sort ascending."""
    seen: dict[int, dict] = {}
    for c in candles:
        seen.setdefault(c["from_ts"], c)
    return [seen[k] for k in sorted(seen)]


def find_gaps(candles: list[dict], interval: int) -> list[dict]:
    """Missing-candle ranges in a sorted, deduped series."""
    gaps = []
    for prev, cur in zip(candles, candles[1:]):
        expected = prev["from_ts"] + interval
        if cur["from_ts"] > expected:
            gaps.append(
                {
                    "after_ts": prev["from_ts"],
                    "resume_ts": cur["from_ts"],
                    "missing": (cur["from_ts"] - expected) // interval,
                }
            )
    return gaps


def all_failed(results: list[dict]) -> bool:
    """True when a collection produced no dataset at all (health signal:
    an empty collection must never look like success to launchd)."""
    return bool(results) and all("error" in r for r in results)


# ---------------- live collection ----------------

def _connect_client():
    import os

    from run_once import _call, _load_env

    _load_env()
    from iqoptionapi.stable_api import IQ_Option

    client = IQ_Option(os.environ["IQ_EMAIL"], os.environ["IQ_PASSWORD"])
    ok, reason = _call(client.connect, timeout=90)
    if not ok:
        raise SystemExit(f"login failed: {reason}")
    return client, _call


def collect_candles(assets: list[str], interval: int, hours: float) -> list[dict]:
    """Collect one immutable dataset per asset over a single broker session.

    One asset failing (closed market, unknown symbol, timeout) never blocks
    the others - its result carries an 'error' entry instead."""
    import pandas as pd

    from storage import open_db, store_dataset
    from validation import validate_candles

    client, _call = _connect_client()
    conn = open_db()
    results = []
    for asset in assets:
        try:
            end_ts = time.time()
            raw: list[dict] = []
            for page_end in plan_pages(end_ts, hours, interval):
                page = _call(client.get_candles, asset, interval, PAGE_SIZE, page_end, timeout=60)
                raw.extend(normalize_candle(c) for c in page)

            cutoff = end_ts - hours * 3600
            candles = [
                c for c in dedupe_candles(raw) if c["from_ts"] >= cutoff and c["to_ts"] <= end_ts
            ]
            gaps = find_gaps(candles, interval)

            # Validation failure aborts this asset - a bad dataset must never be stored.
            validate_candles(pd.DataFrame(candles), interval)

            dataset_id = store_dataset(conn, asset, interval, candles, gaps)
            results.append(
                {
                    "dataset_id": dataset_id,
                    "asset": asset,
                    "interval_seconds": interval,
                    "candles": len(candles),
                    "gaps": len(gaps),
                    "gap_detail": gaps,
                    "start_utc": datetime.fromtimestamp(candles[0]["from_ts"], timezone.utc).isoformat() if candles else None,
                    "end_utc": datetime.fromtimestamp(candles[-1]["to_ts"], timezone.utc).isoformat() if candles else None,
                }
            )
        except Exception as exc:
            results.append({"asset": asset, "error": f"{type(exc).__name__}: {exc}"})
    return results


def collect_payouts() -> dict:
    from storage import open_db, store_payout_snapshot

    client, _call = _connect_client()
    profits = _call(client.get_all_profit, timeout=90)
    conn = open_db()
    count = store_payout_snapshot(conn, {a: dict(k) for a, k in profits.items()})
    return {"payout_rows": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    candles_cmd = sub.add_parser("candles", help="collect historical candles, one dataset per asset")
    candles_cmd.add_argument("assets", nargs="+")
    candles_cmd.add_argument("--interval", type=int, default=60, help="candle seconds (default 60)")
    candles_cmd.add_argument("--hours", type=float, default=24.0, help="lookback hours (default 24)")

    sub.add_parser("payouts", help="snapshot current payout ratios for all assets")

    args = parser.parse_args()
    if args.command == "candles":
        results = collect_candles(args.assets, args.interval, args.hours)
        print(json.dumps(results, indent=2))
        failures = [r for r in results if "error" in r]
        for failure in failures:
            print(f"PARTIAL FAILURE: {failure['asset']}: {failure['error']}", file=sys.stderr)
        if all_failed(results):
            print("ALL ASSETS FAILED - no dataset stored this cycle", file=sys.stderr)
            sys.exit(1)
    else:
        print(json.dumps(collect_payouts(), indent=2))


if __name__ == "__main__":
    main()
