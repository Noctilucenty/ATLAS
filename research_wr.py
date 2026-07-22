"""RESEARCH SCREENING ONLY - win-rate improvement harness.

Decade-scale walk-forward with a strict SELECTION (<= 2022) / HOLDOUT
(>= 2023) split so nothing is judged on the data that chose it. Produces:

  1. LOSS CONCENTRATION - holdout win rate by session, volatility, trend,
     model-confidence, and asset. (items 1, 9)
  6. FEATURE STABILITY - per-fold LightGBM importances; consistently weak
     features are prune candidates. (item 6)
  7. THRESHOLD OPTIMISATION - a grid over (ev_margin, meta_threshold) scored
     on SELECTION, choosing the win-rate-maximising cell subject to a minimum
     holdout-trade floor and positive post-payout EV, then reporting that
     cell's HOLDOUT win rate and significance. (items 2, 7)

Leak-free: the meta model is trained on SELECTION trades only (the deployed
meta-h3.pkl is refit on all years and would leak the holdout). Screening
only - a winning cell must still be pre-registered and forward tested.

Two-phase so pairs compute in parallel:
    python research_wr.py --compute-pair eurusd --dump research_logs/wr_eurusd.pkl
    ... (one process per pair, in parallel) ...
    python research_wr.py --aggregate 'research_logs/wr_*.pkl'
"""

import argparse
import glob
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features import (
    EXTRA_MTF_COLUMNS,
    EXTRA_VOL_COLUMNS,
    FEATURE_COLUMNS,
    build_features,
)

ENRICHED_COLUMNS = list(FEATURE_COLUMNS) + EXTRA_VOL_COLUMNS + EXTRA_MTF_COLUMNS
from research_deephistory import download_years, load_candles
from train import ChronoCalibratedModel, decide_action

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_TS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())

# Context features the leak-free meta model sees (mirrors research_meta).
META_CONTEXT = [
    "p_up", "conf", "is_call",
    "ret_1", "ret_5", "ret_15", "ret_60", "adx", "bb_pctb", "macd_hist_atr",
    "ema_spread_atr", "ema_fast_slope", "rsi", "atr_norm", "body_ratio",
    "vol_regime", "mtf_align", "hour_sin", "hour_cos",
    "session_asia", "session_europe", "session_us",
]


def compute_pair(pair: str, horizon: int, splits: int, cache_dir: str,
                 enrich: bool = False) -> dict:
    """Walk-forward predictions + per-fold base-model importances for one pair.

    enrich=True additionally computes the extra_vol + extra_mtf feature
    blocks (carried in preds for meta-v2 research) and trains a SECOND
    direction model per fold on the enriched set, dumped as p_up_ext, so
    consensus gating can be tested offline. The base p_up stays the
    production feature contract either way."""
    purge_s = horizon * 60
    candles = load_candles(download_years(pair, range(2016, 2026), Path(cache_dir)))
    ff = build_features(candles, interval=60, horizon=horizon, entry_next_open=True,
                        extra_vol=enrich, extra_mtf=enrich)
    ff = ff.dropna(subset=["label_up"]).reset_index(drop=True)
    ts = ff["to_ts"].to_numpy()
    edges = np.linspace(ts[0], ts[-1], splits + 2)

    preds, importances = [], []
    for k in range(1, splits + 1):
        te = np.where((ts > edges[k]) & (ts <= edges[k + 1]))[0]
        tr = np.where(ts <= edges[k] - purge_s)[0]
        if not len(te) or len(tr) < 10000:
            continue
        model = ChronoCalibratedModel(n_folds=3, gap=horizon, model_kind="lgbm")
        model.fit(ff[FEATURE_COLUMNS].iloc[tr], ff["label_up"].iloc[tr])
        base = getattr(model, "base_", None)
        if base is not None and hasattr(base, "feature_importances_"):
            importances.append(dict(zip(FEATURE_COLUMNS, base.feature_importances_)))
        sub = ff.iloc[te].copy()
        sub["p_up"] = model.predict_proba_up(ff[FEATURE_COLUMNS].iloc[te])
        if enrich:
            ext = ChronoCalibratedModel(n_folds=3, gap=horizon, model_kind="lgbm")
            ext.fit(ff[ENRICHED_COLUMNS].iloc[tr], ff["label_up"].iloc[tr])
            sub["p_up_ext"] = ext.predict_proba_up(ff[ENRICHED_COLUMNS].iloc[te])
        sub["asset"] = pair.upper()
        preds.append(sub)
        print(f"  {pair} fold {k}", flush=True)
    return {"pair": pair.upper(),
            "preds": pd.concat(preds, ignore_index=True),
            "importances": importances}


def compute_pooled(pairs: list, horizon: int, splits: int, cache_dir: str) -> dict:
    """Global model: ONE LightGBM per fold trained on ALL pairs pooled, then
    predicting each pair's test rows. Time-purged folds cut on to_ts. Returns
    the same bundle shape as compute_pair so aggregate() compares specialists
    (per-pair models) against this global model at matched thresholds."""
    purge_s = horizon * 60
    parts = []
    for pair in pairs:
        candles = load_candles(download_years(pair, range(2016, 2026), Path(cache_dir)))
        ff = build_features(candles, interval=60, horizon=horizon, entry_next_open=True)
        ff = ff.dropna(subset=["label_up"]).copy()
        ff["asset"] = pair.upper()
        parts.append(ff)
    pooled = pd.concat(parts, ignore_index=True).sort_values("to_ts").reset_index(drop=True)
    ts = pooled["to_ts"].to_numpy()
    edges = np.linspace(ts[0], ts[-1], splits + 2)
    n_assets = pooled["asset"].nunique()

    preds, importances = [], []
    for k in range(1, splits + 1):
        te = np.where((ts > edges[k]) & (ts <= edges[k + 1]))[0]
        tr = np.where(ts <= edges[k] - purge_s)[0]
        if not len(te) or len(tr) < 10000:
            continue
        model = ChronoCalibratedModel(n_folds=3, gap=horizon * n_assets, model_kind="lgbm")
        model.fit(pooled[FEATURE_COLUMNS].iloc[tr], pooled["label_up"].iloc[tr])
        base = getattr(model, "base_", None)
        if base is not None and hasattr(base, "feature_importances_"):
            importances.append(dict(zip(FEATURE_COLUMNS, base.feature_importances_)))
        sub = pooled.iloc[te].copy()
        sub["p_up"] = model.predict_proba_up(pooled[FEATURE_COLUMNS].iloc[te])
        preds.append(sub)
        print(f"  pooled fold {k}", flush=True)
    return {"pair": "POOLED", "preds": pd.concat(preds, ignore_index=True),
            "importances": importances}


def independent(trades: pd.DataFrame, purge_s: int) -> pd.DataFrame:
    keep, last = [], {}
    for row in trades.sort_values("ts").itertuples():
        if row.ts >= last.get(row.asset, -1) + purge_s:
            keep.append(row.Index)
            last[row.asset] = row.ts
    return trades.loc[keep]


def wr_row(won: pd.Series) -> str:
    n = len(won)
    return f"{won.mean():.1%} (n={n})" if n else "-"


def session_of(sec: float) -> str:
    h = sec // 3600
    return "asia" if h < 7 else "europe" if h < 13 else "us" if h < 21 else "late"


def fit_meta_on_selection(base: pd.DataFrame, sel_mask: np.ndarray) -> np.ndarray:
    from lightgbm import LGBMClassifier

    feat = [c for c in META_CONTEXT if c in base.columns]
    for pair in base["asset"].str.replace("-OTC", "", regex=False).unique():
        col = f"pair_{pair}"
        base[col] = (base["asset"].str.replace("-OTC", "", regex=False) == pair).astype(float)
        feat.append(col)
    meta = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, verbosity=-1,
    )
    meta.fit(base.loc[sel_mask, feat], base.loc[sel_mask, "won"])
    return meta.predict_proba(base[feat])[:, 1]


def aggregate(bundles: list, horizon: int, payout: float, min_holdout: int) -> dict:
    purge_s = horizon * 60
    breakeven = 1.0 / (1.0 + payout)
    allrows = pd.concat([b["preds"] for b in bundles], ignore_index=True)
    importances = [imp for b in bundles for imp in b["importances"]]

    # ---- feature stability (item 6) ----
    imp_df = pd.DataFrame(importances)
    stability = {
        f: {"mean": round(float(imp_df[f].mean()), 1),
            "std": round(float(imp_df[f].std()), 1),
            "min": int(imp_df[f].min())}
        for f in FEATURE_COLUMNS
    }
    stability = dict(sorted(stability.items(), key=lambda kv: -kv[1]["mean"]))
    med = np.median([s["mean"] for s in stability.values()])
    weak = [f for f, v in stability.items() if v["mean"] < 0.5 * med]

    # ---- trade table at the loosest gate, with context ----
    base = allrows.copy()
    base["action"] = [decide_action(float(p), payout, 0.0) for p in base["p_up"]]
    base = base[base["action"] != "no_trade"].reset_index(drop=True)
    base["won"] = ((base["label_up"] == 1.0) == (base["action"] == "binary_call")).astype(float)
    base["is_call"] = (base["action"] == "binary_call").astype(float)
    base["conf"] = (base["p_up"] - 0.5).abs()
    ce = base["p_up"] * payout - (1 - base["p_up"])
    pe = (1 - base["p_up"]) * payout - base["p_up"]
    base["ev"] = np.maximum(ce, pe)
    base["ts"] = base["to_ts"].astype(int)
    base["session"] = (base["to_ts"] % 86400).map(session_of)
    base["vol_bucket"] = pd.cut(base["vol_regime"], [0, .33, .67, 1.01], labels=["low", "mid", "high"])
    base["adx_bucket"] = pd.cut(base["adx"], [-.01, .2, .3, 1.01], labels=["weak", "mid", "strong"])
    base["conf_bucket"] = pd.cut(base["conf"], [0, .02, .04, .06, .5], labels=["0-2", "2-4", "4-6", "6+"])

    sel_mask = (base["to_ts"] < SPLIT_TS).to_numpy()
    base["meta_p"] = fit_meta_on_selection(base, sel_mask)   # leak-free
    hold = base[~sel_mask]
    sel = base[sel_mask]

    # ---- loss concentration on holdout (items 1, 9) ----
    hold_ind = independent(hold, purge_s)
    concentration = {
        dim: {str(k): wr_row(g["won"]) for k, g in hold_ind.groupby(dim, observed=True)}
        for dim in ["asset", "session", "vol_bucket", "adx_bucket", "conf_bucket"]
    }

    # ---- threshold optimisation: pick on selection, report on holdout ----
    grid = []
    for ev_m in (0.02, 0.03, 0.04, 0.05, 0.06):
        for meta_t in (0.0, 0.50, 0.55, 0.60, 0.65, 0.70):
            s = independent(sel[(sel["ev"] > ev_m) & (sel["meta_p"] >= meta_t)], purge_s)
            if len(s) < 50:
                continue
            grid.append({"ev_margin": ev_m, "meta_threshold": meta_t,
                         "sel_trades": len(s), "sel_wr": round(float(s["won"].mean()), 4)})
    ranked = sorted(grid, key=lambda c: -c["sel_wr"])
    winner = None
    for c in ranked:
        h = independent(hold[(hold["ev"] > c["ev_margin"]) & (hold["meta_p"] >= c["meta_threshold"])], purge_s)
        if len(h) >= min_holdout and c["sel_wr"] > breakeven:
            p = stats.binomtest(int(h["won"].sum()), len(h), breakeven, alternative="greater").pvalue
            winner = {**c, "holdout_trades": len(h), "holdout_wr": round(float(h["won"].mean()), 4),
                      "breakeven": round(breakeven, 4), "p_beats_breakeven": round(float(p), 6)}
            break

    b = independent(hold[(hold["ev"] > 0.03) & (hold["meta_p"] >= 0.60)], purge_s)
    baseline = {"ev_margin": 0.03, "meta_threshold": 0.60, "holdout_trades": len(b),
                "holdout_wr": round(float(b["won"].mean()), 4) if len(b) else None}

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon, "payout": payout, "breakeven": round(breakeven, 4),
        "pairs": [b["pair"] for b in bundles],
        "feature_stability": stability,
        "weak_feature_candidates": weak,
        "loss_concentration_holdout": concentration,
        "current_baseline_holdout": baseline,
        "optimised_winner_holdout": winner,
        "threshold_grid_selection_top": ranked[:12],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute-pooled", action="store_true",
                    help="global model: train pooled over --pairs, pickle to --dump")
    ap.add_argument("--compute-pair", default=None, help="compute one pair and pickle to --dump")
    ap.add_argument("--enrich", action="store_true",
                    help="carry extra_vol+extra_mtf columns and a second enriched direction model (p_up_ext)")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--aggregate", default=None, help="glob of pair pickles to analyse")
    ap.add_argument("--pairs", default="eurusd,gbpusd,usdjpy", help="fallback single-process mode")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--payout", type=float, default=0.87)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--min-holdout-trades", type=int, default=200)
    ap.add_argument("--cache-dir", default="histdata_cache")
    ap.add_argument("--out", default="research_logs/wr_report.json")
    args = ap.parse_args()

    if args.compute_pooled:
        bundle = compute_pooled(args.pairs.split(","), args.horizon, args.splits, args.cache_dir)
        with open(args.dump, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"dumped POOLED -> {args.dump}")
        return

    if args.compute_pair:
        bundle = compute_pair(args.compute_pair, args.horizon, args.splits,
                              args.cache_dir, enrich=args.enrich)
        with open(args.dump, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"dumped {args.compute_pair} -> {args.dump}")
        return

    if args.aggregate:
        bundles = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(args.aggregate))]
    else:  # single-process fallback
        bundles = [compute_pair(p, args.horizon, args.splits, args.cache_dir)
                   for p in args.pairs.split(",")]

    report = aggregate(bundles, args.horizon, args.payout, args.min_holdout_trades)
    (PROJECT_DIR / args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ["weak_feature_candidates", "current_baseline_holdout",
                       "optimised_winner_holdout"]}, indent=2))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
