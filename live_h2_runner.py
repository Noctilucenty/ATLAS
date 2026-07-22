"""Live hypothesis-#2 runner - PAPER MODE BY DEFAULT.

Each minute: fetch fresh 1m candles for every registered instrument, build
v1.3.0 features + cross-asset currency-strength columns for the just-closed
bar, apply the frozen H2 model (models/h2-*.pkl, newest), gate by expected
value against the LIVE binary payout, and append one JSON line per signal
to logs/live_h2.jsonl.

--trade places a $1 PRACTICE 15-minute binary per signal (hard PRACTICE
guard, execution refused otherwise). Default is paper: nothing is placed;
outcomes are scored later against collected candles. Run bounded with
--minutes; schedule externally for longer sessions.
"""

import argparse
import json
import os
import pickle
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features
from instruments import INSTRUMENTS
from research_pooled import currencies
from train import decide_action

PROJECT_DIR = Path(__file__).resolve().parent
CANDLE_COUNT = 560  # covers REGIME_WINDOW=480 warmup plus slack
EV_MARGIN = 0.03    # FORWARD_TEST.md hypothesis #2 primary gate - do not tune
EXPIRY_MINUTES = 15
TRADE_AMOUNT = 1.0


def _call(fn, *args, timeout=60, **kwargs):
    """Blocking library call on a daemon thread with a hard timeout (the
    iqoptionapi library busy-waits forever on lost replies)."""
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


def load_frozen_model():
    paths = sorted((PROJECT_DIR / "models").glob("h2-*.pkl"))
    if not paths:
        raise SystemExit("no models/h2-*.pkl - run live_model_build.py first")
    with open(paths[-1], "rb") as fh:
        bundle = pickle.load(fh)
    return paths[-1].name, bundle["model"], bundle["meta"]


def load_meta_filter():
    """Optional hypothesis-#3 meta model. Signals are NEVER gated by it here;
    its probability is logged per signal so H3 can be evaluated later."""
    path = PROJECT_DIR / "models" / "meta-h3.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def meta_probability(meta_bundle, row, p: float, action: str) -> float | None:
    if meta_bundle is None:
        return None
    feats = {
        "p_up": p, "conf": abs(p - 0.5),
        "is_call": float(action == "binary_call"),
    }
    for col in meta_bundle["features"]:
        if col in feats:
            continue
        if col.startswith("pair_"):
            feats[col] = float(col == f"pair_{row['asset'].replace('-OTC', '')}")
        else:
            feats[col] = float(row[col]) if col in row.index else 0.0
    frame = pd.DataFrame([feats])[meta_bundle["features"]]
    return float(meta_bundle["model"].predict_proba(frame)[0, 1])


def normalize(raw: list) -> pd.DataFrame:
    rows = [
        {
            "from_ts": int(c["from"]), "to_ts": int(c["to"]),
            "open": float(c["open"]), "high": float(c["max"]),
            "low": float(c["min"]), "close": float(c["close"]),
            "volume": float(c.get("volume") or 0.0),
        }
        for c in raw
    ]
    return pd.DataFrame(rows).drop_duplicates("from_ts").sort_values("from_ts")


def latest_feature_rows(client, horizon: int) -> pd.DataFrame:
    """One feature row per asset for the most recent CLOSED bar, with
    cross-asset currency-strength columns computed from the same snapshot."""
    now = time.time()
    per_asset = []
    for asset in INSTRUMENTS:
        try:
            raw = _call(client.get_candles, asset, 60, CANDLE_COUNT, now, timeout=30)
            candles = normalize(raw)
            # The newest bar may still be forming; keep only closed bars.
            candles = candles[candles["to_ts"] <= now + 1]
            ff = build_features(candles, interval=60, horizon=horizon)
            if ff.empty:
                continue
            row = ff.iloc[[-1]].copy()
            row["asset"] = asset
            per_asset.append(row)
        except Exception as exc:
            print(f"WARN {asset}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if not per_asset:
        return pd.DataFrame()
    frame = pd.concat(per_asset, ignore_index=True)
    # Cross-asset strengths at this single timestamp, own pair excluded.
    ret5 = dict(zip(frame["asset"], frame["ret_5"]))
    num, cnt = {}, {}
    for asset, r in ret5.items():
        base, quote = currencies(asset)
        for cur, sign in ((base, 1.0), (quote, -1.0)):
            num[cur] = num.get(cur, 0.0) + sign * r
            cnt[cur] = cnt.get(cur, 0) + 1
    base_str, quote_str = [], []
    for asset in frame["asset"]:
        base, quote = currencies(asset)
        r = ret5[asset]
        b_n, b_c = num[base] - r, cnt[base] - 1
        q_n, q_c = num[quote] + r, cnt[quote] - 1
        base_str.append(b_n / b_c if b_c else 0.0)
        quote_str.append(q_n / q_c if q_c else 0.0)
    frame["xs_base_str"] = base_str
    frame["xs_quote_str"] = quote_str
    frame["xs_mkt_vol"] = float(np.mean([abs(r) for r in ret5.values()]))
    return frame


def live_quotes(client) -> dict:
    """Payout table only. get_all_open_time is NOT used: the vendored
    library's digital-options sub-fetch crashes internally and the call
    times out every time. Openness is inferred from candle freshness
    instead - a market that closed a bar in the last 3 minutes is trading,
    and a closed market cannot produce fresh bars."""
    return _call(client.get_all_profit, timeout=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=60, help="how long to run")
    parser.add_argument("--trade", action="store_true",
                        help="place $1 PRACTICE binaries (default: paper log only)")
    args = parser.parse_args()

    _load_env()
    from iqoptionapi.stable_api import IQ_Option

    model_name, model, meta = load_frozen_model()
    meta_bundle = load_meta_filter()
    horizon = meta["horizon_bars"]
    feature_cols = meta["feature_columns"]
    print(f"model={model_name} rows={meta['rows']} data_end={meta['data_end_ts']}", flush=True)

    client = IQ_Option(os.environ["IQ_EMAIL"], os.environ["IQ_PASSWORD"])
    ok, reason = _call(client.connect, timeout=90)
    if not ok:
        raise SystemExit(f"login failed: {reason}")
    _call(client.change_balance, "PRACTICE")
    if _call(client.get_balance_mode) != "PRACTICE":
        raise SystemExit("refusing to run: balance mode is not PRACTICE")

    log_path = PROJECT_DIR / "logs" / "live_h2.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    deadline = time.time() + args.minutes * 60
    open_orders = []

    while time.time() < deadline:
        # Wait for the next minute boundary + 2s so the bar is fully closed.
        time.sleep(max(0.0, 60 - time.time() % 60) + 2)
        cycle_ts = int(time.time())
        try:
            frame = latest_feature_rows(client, horizon)
            if frame.empty:
                continue
            profits = live_quotes(client)
            p_up = model.predict_proba_up(frame[feature_cols])
            fired = 0
            for (_, row), p in zip(frame.iterrows(), p_up):
                asset = row["asset"]
                spec = INSTRUMENTS[asset]
                quote = profits.get(spec.quote_key, {})
                payout = quote.get("binary") or quote.get(spec.option_kind)
                is_open = (cycle_ts - int(row["to_ts"])) < 180  # fresh bar = trading
                if not isinstance(payout, (int, float)) or not is_open:
                    continue
                action = decide_action(float(p), float(payout), EV_MARGIN)
                if action == "no_trade":
                    continue
                meta_p = meta_probability(meta_bundle, row, float(p), action)
                record = {
                    "ts": cycle_ts,
                    "bar_to_ts": int(row["to_ts"]),
                    "asset": asset,
                    "action": action,
                    "p_up": round(float(p), 6),
                    "meta_p": round(meta_p, 4) if meta_p is not None else None,
                    "payout": float(payout),
                    "ev_margin": EV_MARGIN,
                    "model": model_name,
                    "mode": "trade" if args.trade else "paper",
                }
                if args.trade:
                    ok, order_id = _call(
                        client.buy, TRADE_AMOUNT, spec.order_active,
                        "call" if action == "binary_call" else "put",
                        EXPIRY_MINUTES, timeout=60,
                    )
                    record["order_id"] = order_id if ok else None
                    if not ok:
                        record["order_error"] = str(order_id)
                    else:
                        open_orders.append((order_id, record))
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                fired += 1
            # Heartbeat: an idle cycle and a broken one look identical in the
            # signal log, so record that the model really did evaluate.
            with open(log_path.with_name("live_h2_heartbeat.jsonl"), "a") as fh:
                fh.write(json.dumps({
                    "ts": cycle_ts,
                    "assets": len(frame),
                    "max_conf": round(float(np.max(np.abs(p_up - 0.5))), 4),
                    "signals": fired,
                }) + "\n")
        except Exception as exc:
            print(f"WARN cycle: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    # Collect outcomes for any orders still open before exiting.
    for order_id, record in open_orders:
        try:
            result, profit = _call(client.check_win_v4, order_id, timeout=EXPIRY_MINUTES * 60 + 120)
            outcome = {**record, "result": result, "profit": profit, "settled": True}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(outcome) + "\n")
        except Exception as exc:
            print(f"WARN settle {order_id}: {exc}", file=sys.stderr, flush=True)
    print(f"done {datetime.now(timezone.utc).isoformat()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
