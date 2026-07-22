"""RESEARCH SCREENING ONLY - win-rate improvement harness.

One decade-scale walk-forward pass that produces, with a strict
SELECTION (<= 2022) / HOLDOUT (>= 2023) split so nothing is judged on the
data that chose it:

  1. LOSS CONCENTRATION - holdout win rate by session, volatility bucket,
     trend strength, model-confidence bucket, and asset. (items 1, 9)
  6. FEATURE STABILITY - each base feature's LightGBM importance across every
     walk-forward fold; features that are consistently weak are prune
     candidates. (item 6)
  7. THRESHOLD OPTIMISATION - a grid over (ev_margin, meta_threshold) scored
     on SELECTION, choosing the win-rate-maximising cell subject to a minimum
     independent-trade floor and positive post-payout EV, then reporting that
     exact cell's HOLDOUT win rate, trade count and significance. (items 2, 7)

Uses the production calibrated model (ChronoCalibratedModel, lgbm) for gating
probabilities and the deployed meta model (models/meta-h3.pkl) for meta_p, so
any chosen threshold transfers to live use. Writes NO run bundle and never
feeds execution - a winning cell must still be pre-registered and forward
tested before it changes anything.
"""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features import FEATURE_COLUMNS, build_features
from research_deephistory import download_years, load_candles
from train import ChronoCalibratedModel, _prob_up, decide_action

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_TS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())


def load_meta():
    path = PROJECT_DIR / "models" / "meta-h3.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def meta_prob(meta, frame: pd.DataFrame, p_up, actions) -> np.ndarray:
    if meta is None:
        return np.full(len(frame), np.nan)
    feats = pd.DataFrame(index=frame.index)
    for col in meta["features"]:
        if col == "p_up":
            feats[col] = p_up
        elif col == "conf":
            feats[col] = np.abs(np.asarray(p_up) - 0.5)
        elif col == "is_call":
            feats[col] = [float(a == "binary_call") for a in actions]
        elif col.startswith("pair_"):
            base = col[len("pair_"):]
            feats[col] = (frame["asset"].str.replace("-OTC", "", regex=False) == base).astype(float)
        else:
            feats[col] = frame[col].to_numpy() if col in frame.columns else 0.0
    return meta["model"].predict_proba(feats[meta["features"]])[:, 1]


def independent(trades: pd.DataFrame, purge_s: int) -> pd.DataFrame:
    """Keep, per asset, only trades whose window does not overlap the previous
    kept trade of that asset."""
    keep, last = [], {}
    for row in trades.sort_values("ts").itertuples():
        if row.ts >= last.get(row.asset, -1) + purge_s:
            keep.append(row.Index)
            last[row.asset] = row.ts
    return trades.loc[keep]


def wr_row(won: pd.Series) -> str:
    n = len(won)
    return f"{won.mean():.1%} (n={n})" if n else "-"


def session_of(sec_of_day: float) -> str:
    h = sec_of_day // 3600
    return "asia" if h < 7 else "europe" if h < 13 else "us" if h < 21 else "late"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="eurusd,gbpusd,usdjpy")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--payout", type=float, default=0.87)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--min-holdout-trades", type=int, default=200,
                    help="threshold sweep floor: independent holdout trades required")
    ap.add_argument("--cache-dir", default="histdata_cache")
    ap.add_argument("--out", default="research_logs/wr_report.json")
    args = ap.parse_args()

    purge_s = args.horizon * 60
    breakeven = 1.0 / (1.0 + args.payout)
    meta = load_meta()

    rows, importances = [], []
    for pair in args.pairs.split(","):
        candles = load_candles(download_years(pair, range(2016, 2026), Path(args.cache_dir)))
        ff = build_features(candles, interval=60, horizon=args.horizon, entry_next_open=True)
        ff = ff.dropna(subset=["label_up"]).reset_index(drop=True)
        ts = ff["to_ts"].to_numpy()
        edges = np.linspace(ts[0], ts[-1], args.splits + 2)
        for k in range(1, args.splits + 1):
            te = np.where((ts > edges[k]) & (ts <= edges[k + 1]))[0]
            tr = np.where(ts <= edges[k] - purge_s)[0]
            if not len(te) or len(tr) < 10000:
                continue
            model = ChronoCalibratedModel(n_folds=3, gap=args.horizon, model_kind="lgbm")
            model.fit(ff[FEATURE_COLUMNS].iloc[tr], ff["label_up"].iloc[tr])
            # base_ is the LGBM for model_kind="lgbm"; capture its importances.
            imp = getattr(model, "base_", None)
            if imp is not None and hasattr(imp, "feature_importances_"):
                importances.append(dict(zip(FEATURE_COLUMNS, imp.feature_importances_)))
            p_up = model.predict_proba_up(ff[FEATURE_COLUMNS].iloc[te])
            sub = ff.iloc[te].copy()
            sub["p_up"] = p_up
            sub["asset"] = pair.upper()
            rows.append(sub)
        print(f"{pair}: folds done", flush=True)

    allrows = pd.concat(rows, ignore_index=True)

    # ---- feature stability (item 6) ----
    imp_df = pd.DataFrame(importances)
    stability = {
        f: {"mean": round(float(imp_df[f].mean()), 1),
            "std": round(float(imp_df[f].std()), 1),
            "min": int(imp_df[f].min())}
        for f in FEATURE_COLUMNS
    }
    stability = dict(sorted(stability.items(), key=lambda kv: -kv[1]["mean"]))
    weak = [f for f, v in stability.items()
            if v["mean"] < 0.5 * np.median([s["mean"] for s in stability.values()])]

    # ---- build the trade table once at the loosest gate, tag context ----
    base = allrows.copy()
    base["action"] = [decide_action(float(p), args.payout, 0.0) for p in base["p_up"]]
    base = base[base["action"] != "no_trade"].reset_index(drop=True)
    base["won"] = ((base["label_up"] == 1.0) == (base["action"] == "binary_call")).astype(float)
    base["conf"] = (base["p_up"] - 0.5).abs()
    base["call_ev"] = base["p_up"] * args.payout - (1 - base["p_up"])
    base["put_ev"] = (1 - base["p_up"]) * args.payout - base["p_up"]
    base["ev"] = base[["call_ev", "put_ev"]].max(axis=1)
    base["meta_p"] = meta_prob(meta, base, base["p_up"].to_numpy(),
                               base["action"].tolist())
    base["session"] = (base["to_ts"] % 86400).map(session_of)
    base["vol_bucket"] = pd.cut(base["vol_regime"], [0, .33, .67, 1.01],
                                labels=["low", "mid", "high"])
    base["adx_bucket"] = pd.cut(base["adx"], [-.01, .2, .3, 1.01],
                                labels=["weak", "mid", "strong"])
    base["conf_bucket"] = pd.cut(base["conf"], [0, .02, .04, .06, .5],
                                 labels=["0-2", "2-4", "4-6", "6+"])
    base["ts"] = base["to_ts"].astype(int)

    hold = base[base["to_ts"] >= SPLIT_TS]
    sel = base[base["to_ts"] < SPLIT_TS]

    # ---- loss concentration on HOLDOUT independent trades (items 1, 9) ----
    hold_ind = independent(hold, purge_s)
    concentration = {
        dim: {str(k): wr_row(g["won"]) for k, g in hold_ind.groupby(dim, observed=True)}
        for dim in ["asset", "session", "vol_bucket", "adx_bucket", "conf_bucket"]
    }

    # ---- threshold optimisation: choose on SELECTION, report on HOLDOUT ----
    grid = []
    for ev_m in (0.02, 0.03, 0.04, 0.05, 0.06):
        for meta_t in (0.0, 0.50, 0.55, 0.60, 0.65, 0.70):
            s = independent(sel[(sel["ev"] > ev_m) & (sel["meta_p"] >= meta_t)], purge_s)
            if len(s) < 50:
                continue
            grid.append({"ev_margin": ev_m, "meta_threshold": meta_t,
                         "sel_trades": len(s), "sel_wr": float(s["won"].mean())})
    # winner: max selection WR with enough projected holdout volume and +EV.
    hold_n = len(independent(hold, purge_s))
    ranked = sorted(grid, key=lambda c: -c["sel_wr"])
    winner = None
    for c in ranked:
        h = independent(hold[(hold["ev"] > c["ev_margin"]) & (hold["meta_p"] >= c["meta_threshold"])], purge_s)
        if len(h) >= args.min_holdout_trades and c["sel_wr"] > breakeven:
            wr = float(h["won"].mean())
            p = stats.binomtest(int(h["won"].sum()), len(h), breakeven, alternative="greater").pvalue
            winner = {**c, "holdout_trades": len(h), "holdout_wr": round(wr, 4),
                      "breakeven": round(breakeven, 4),
                      "p_beats_breakeven": round(float(p), 5)}
            break

    # baseline (current deployed gate) for comparison, on holdout.
    b = independent(hold[(hold["ev"] > 0.03) & (hold["meta_p"] >= 0.60)], purge_s)
    baseline = {"ev_margin": 0.03, "meta_threshold": 0.60, "holdout_trades": len(b),
                "holdout_wr": round(float(b["won"].mean()), 4) if len(b) else None}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": args.pairs, "horizon": args.horizon, "payout": args.payout,
        "breakeven": round(breakeven, 4),
        "holdout_independent_trades_ungated": hold_n,
        "feature_stability": stability,
        "weak_feature_candidates": weak,
        "loss_concentration_holdout": concentration,
        "current_baseline_holdout": baseline,
        "optimised_winner_holdout": winner,
        "threshold_grid_selection": ranked[:10],
    }
    out = PROJECT_DIR / args.out
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ["weak_feature_candidates", "current_baseline_holdout",
                       "optimised_winner_holdout"]}, indent=2))
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
