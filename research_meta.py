"""RESEARCH SCREENING ONLY - meta-labeling + gating tables on decade trades.

Consumes the prediction dumps from research_deephistory --dump-preds, joins
each gated trade back to its full feature context, then answers two
questions with decade-scale samples:

1. Gating tables: independent-trade win rate by session / volatility regime
   / trend strength, split into a SELECTION period (2016-2022) and a
   HOLDOUT period (2023-2025) so any rule read off the tables is honestly
   scored on years it never saw.
2. Meta-labeling: an LGBM trained on selection-period trades to predict
   "will this gated trade win?", evaluated on holdout trades at several
   acceptance thresholds. Reports win rate + trade retention per threshold.

No run bundles; never feeds execution.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features import build_features
from research_deephistory import download_years, load_candles
from train import decide_action

CONTEXT_COLS = [
    "p_up", "conf", "is_call",
    "ret_1", "ret_5", "ret_15", "ret_60",
    "adx", "bb_pctb", "macd_hist_atr", "ema_spread_atr", "ema_fast_slope",
    "rsi", "atr_norm", "body_ratio", "vol_regime", "mtf_align",
    "hour_sin", "hour_cos", "session_asia", "session_europe", "session_us",
]
SPLIT_TS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())


def trade_table(pair: str, preds_path: Path, horizon: int, payout: float,
                margin: float, cache_dir: str) -> pd.DataFrame:
    preds = json.load(open(preds_path))
    zips = download_years(pair, range(2016, 2026), Path(cache_dir))
    candles = load_candles(zips)
    ff = build_features(candles, interval=60, horizon=horizon, entry_next_open=True)
    ctx = ff.set_index(ff["to_ts"].astype(int))

    rows, last = [], -1
    purge_s = horizon * 60
    for _, ts, p, label in sorted(preds, key=lambda r: r[1]):
        action = decide_action(float(p), payout, margin)
        if action == "no_trade" or ts < last + purge_s or ts not in ctx.index:
            continue
        last = ts
        c = ctx.loc[ts]
        rows.append({
            "pair": pair.upper(), "ts": ts,
            "p_up": float(p), "conf": abs(float(p) - 0.5),
            "is_call": float(action == "binary_call"),
            "won": float((label == 1.0) == (action == "binary_call")),
            **{k: float(c[k]) for k in CONTEXT_COLS if k in c.index},
        })
    return pd.DataFrame(rows)


def wr_table(df: pd.DataFrame, key) -> dict:
    out = {}
    for name, grp in df.groupby(key):
        out[str(name)] = f"{grp['won'].mean():.1%} (n={len(grp)})"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds-dir", required=True)
    parser.add_argument("--pairs", default="eurusd,gbpusd,usdjpy")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--payout", type=float, default=0.87)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--cache-dir", default="histdata_cache")
    parser.add_argument("--save-model", default=None,
                        help="after holdout reporting, refit the meta model on ALL "
                        "trades and pickle {model, features, meta} to this path")
    args = parser.parse_args()

    frames = []
    for pair in args.pairs.split(","):
        path = Path(args.preds_dir) / f"preds_{pair}.json"
        t = trade_table(pair, path, args.horizon, args.payout, args.margin, args.cache_dir)
        print(f"{pair}: {len(t)} independent gated trades", flush=True)
        frames.append(t)
    trades = pd.concat(frames, ignore_index=True)
    for pair in trades["pair"].unique():
        trades[f"pair_{pair}"] = (trades["pair"] == pair).astype(float)

    sel = trades[trades["ts"] < SPLIT_TS]
    hold = trades[trades["ts"] >= SPLIT_TS]
    print(f"\nselection 2016-22: {len(sel)} trades, wr={sel['won'].mean():.1%}")
    print(f"holdout   2023-25: {len(hold)} trades, wr={hold['won'].mean():.1%}")

    def session_of(r):
        if r["session_asia"]: return "asia"
        if r["session_europe"]: return "europe"
        if r["session_us"]: return "us"
        return "late"

    for name, frame in (("SELECTION", sel), ("HOLDOUT", hold)):
        frame = frame.copy()
        frame["session"] = frame.apply(session_of, axis=1)
        frame["vol_bucket"] = pd.cut(frame["vol_regime"], [0, .33, .67, 1],
                                     labels=["low", "mid", "high"])
        frame["adx_bucket"] = pd.cut(frame["adx"], [0, .2, .3, 1],
                                     labels=["weak", "mid", "strong"])
        print(f"\n[{name}] by session:", json.dumps(wr_table(frame, "session")))
        print(f"[{name}] by vol_regime:", json.dumps(wr_table(frame, "vol_bucket")))
        print(f"[{name}] by adx:", json.dumps(wr_table(frame, "adx_bucket")))
        print(f"[{name}] by pair:", json.dumps(wr_table(frame, "pair")))

    # ---- meta-labeling ----
    from lightgbm import LGBMClassifier

    feat = [c for c in CONTEXT_COLS if c in trades.columns] + [
        c for c in trades.columns if c.startswith("pair_")
    ]
    meta = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, verbosity=-1,
    )
    meta.fit(sel[feat], sel["won"])
    hold = hold.copy()
    hold["meta_p"] = meta.predict_proba(hold[feat])[:, 1]

    base_n, base_wr = len(hold), hold["won"].mean()
    report = []
    for thr in (0.50, 0.55, 0.60, 0.65):
        acc = hold[hold["meta_p"] >= thr]
        if len(acc) < 30:
            continue
        pval = stats.binomtest(int(acc["won"].sum()), len(acc), base_wr).pvalue
        report.append({
            "meta_threshold": thr,
            "trades_kept": f"{len(acc)}/{base_n}",
            "win_rate": round(float(acc["won"].mean()), 4),
            "lift_vs_base": round(float(acc["won"].mean() - base_wr), 4),
            "p_vs_base": round(float(pval), 4),
        })
    print("\nMETA-LABELING (holdout 2023-25, base wr "
          f"{base_wr:.1%} on {base_n}):")
    print(json.dumps(report, indent=2))
    imp = sorted(zip(feat, meta.feature_importances_), key=lambda t: -t[1])[:8]
    print("top meta features:", [f"{n}:{int(v)}" for n, v in imp])

    if args.save_model:
        import pickle

        final = LGBMClassifier(**meta.get_params())
        final.fit(trades[feat], trades["won"])
        payload = {
            "model": final,
            "features": feat,
            "meta": {
                "trained_on": "histdata 2016-2025 gated trades",
                "n_trades": len(trades),
                "holdout_report": report,
                "pairs": sorted(trades["pair"].unique().tolist()),
            },
        }
        with open(args.save_model, "wb") as fh:
            pickle.dump(payload, fh)
        print("saved meta model:", args.save_model)


if __name__ == "__main__":
    main()
