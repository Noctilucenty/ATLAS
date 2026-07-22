"""RESEARCH SCREENING ONLY - decade-scale POOLED validation with
cross-asset currency-strength features (hypothesis #2's key ingredient)
on histdata.com spot bars.

Pools several majors/crosses over years of 1-minute data, adds the same
xs_* features research_pooled uses, and runs the identical time-purged
walk-forward + margin sweep. If the xs features carry signal, the pooled
decade win rate should beat the single-pair decade runs.

Heavy: millions of rows. Bound with --from-year. No run bundles.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from features import FEATURE_COLUMNS, build_features
from research_deephistory import download_years, load_candles
from research_pooled import XS_COLUMNS, add_cross_asset, evaluate_margin, time_folds
from train import ChronoCalibratedModel

DEFAULT_PAIRS = "eurusd,gbpusd,usdjpy,audusd,eurjpy,eurgbp,gbpjpy,usdchf,nzdusd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--from-year", type=int, default=2020)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--payout", type=float, default=0.87)
    parser.add_argument("--ev-margins", default="0.02,0.03,0.04")
    parser.add_argument("--no-xs", action="store_true",
                        help="ablation: drop the cross-asset features")
    parser.add_argument("--cache-dir", default="histdata_cache")
    args = parser.parse_args()

    parts = []
    for pair in args.pairs.split(","):
        zips = download_years(pair, range(args.from_year, args.to_year + 1),
                              Path(args.cache_dir))
        candles = load_candles(zips)
        ff = build_features(candles, interval=60, horizon=args.horizon,
                            entry_next_open=True)
        ff["asset"] = pair.upper()
        parts.append(ff)
        print(f"{pair}: {len(ff)} feature rows", flush=True)
    pooled = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["label_up"])
        .sort_values("to_ts", kind="stable")
        .reset_index(drop=True)
    )
    del parts
    feature_cols = list(FEATURE_COLUMNS)
    if not args.no_xs:
        pooled = add_cross_asset(pooled)
        feature_cols += XS_COLUMNS
    n_assets = pooled["asset"].nunique()
    print(f"pooled rows={len(pooled)} assets={n_assets} "
          f"features={len(feature_cols)}", flush=True)

    purge_s = args.horizon * 60
    X, y = pooled[feature_cols], pooled["label_up"]
    briers, preds = [], []
    for fold, (tr, te) in enumerate(time_folds(pooled, args.splits, purge_s)):
        model = ChronoCalibratedModel(
            n_folds=3, gap=args.horizon * n_assets, model_kind="lgbm"
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        p_up = model.predict_proba_up(X.iloc[te])
        brier = float(brier_score_loss(y.iloc[te].to_numpy(), p_up))
        briers.append(brier)
        rows = pooled.iloc[te]
        preds.extend(zip(rows["asset"], rows["to_ts"].astype(int),
                         map(float, p_up), rows["label_up"]))
        print(f"fold {fold}: brier={brier:.5f} test_rows={len(te)}", flush=True)

    print("mean brier:", round(float(np.mean(briers)), 5))
    results = [
        evaluate_margin(preds, float(m), args.payout, purge_s)
        for m in args.ev_margins.split(",")
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
