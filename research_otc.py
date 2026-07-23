"""RESEARCH - does the edge exist on OTC instruments, and does the deployed
meta model transfer to the broker's instrument set?

Uses ONLY pre-cutoff broker data (to_ts < 2026-07-22 00:00Z) - the forward
window is untouched. Walk-forward over the pooled 28-instrument canonical
history (time-purged folds, production feature contract + cross-asset
columns), EV gate 0.03, then two splits the project has never looked at:

  spot vs OTC   the decade validations were spot-only; 12+ of 28 live
                instruments are broker-synthesised OTC prices.
  meta buckets  meta_p from the DEPLOYED meta-h3.pkl (trained on 3 spot
                pairs) - if win rate rises with meta_p on broker trades,
                the filter transfers; if flat, the H3 hypothesis rests on
                an untransferred model.

Caveat recorded up front: this window is research-contaminated (the H2
config was developed on it), so LEVELS are optimistic - but the SPLITS
(OTC vs spot, meta-bucket slopes) are new questions the contamination has
no reason to bias in either direction.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forward_eval import meta_probabilities
from research_pooled import add_cross_asset, load_pooled, time_folds
from research_wr import independent
from storage import open_db
from train import ChronoCalibratedModel, decide_action
from features import FEATURE_COLUMNS
from research_pooled import XS_COLUMNS

CUTOFF_TS = int(datetime(2026, 7, 22, tzinfo=timezone.utc).timestamp())
PAYOUT = 0.87
EV_MARGIN = 0.03
PURGE_S = 15 * 60


def main() -> None:
    pooled = load_pooled(open_db(), interval=60, horizon=15, entry_next_open=True)
    pooled = add_cross_asset(pooled)
    pooled = pooled[pooled["to_ts"] < CUTOFF_TS].reset_index(drop=True)
    feature_cols = list(FEATURE_COLUMNS) + XS_COLUMNS
    n_assets = pooled["asset"].nunique()
    print(f"pre-cutoff rows={len(pooled)} assets={n_assets}", flush=True)

    preds = []
    for fold, (tr, te) in enumerate(time_folds(pooled, 5, PURGE_S)):
        model = ChronoCalibratedModel(n_folds=3, gap=15 * n_assets, model_kind="lgbm")
        model.fit(pooled[feature_cols].iloc[tr], pooled["label_up"].iloc[tr])
        sub = pooled.iloc[te].copy()
        sub["p_up"] = model.predict_proba_up(pooled[feature_cols].iloc[te])
        preds.append(sub)
        print(f"fold {fold} done", flush=True)
    allp = pd.concat(preds, ignore_index=True)

    base = allp.copy()
    base["action"] = [decide_action(float(p), PAYOUT, EV_MARGIN) for p in base["p_up"]]
    base = base[base["action"] != "no_trade"].reset_index(drop=True)
    base["won"] = ((base["label_up"] == 1.0) == (base["action"] == "binary_call")).astype(float)
    base["ts"] = base["to_ts"].astype(int)
    base["is_otc"] = base["asset"].str.endswith("-OTC")

    with open("models/meta-h3.pkl", "rb") as fh:
        meta_bundle = pickle.load(fh)
    base["meta_p"] = meta_probabilities(
        meta_bundle, base, base["p_up"].to_numpy(), base["action"].tolist()
    )

    def stats_of(df: pd.DataFrame) -> dict:
        ind = independent(df, PURGE_S)
        n = len(ind)
        return {"raw": len(df), "independent": n,
                "wr": round(float(ind["won"].mean()), 4) if n else None}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": "broker canonical history < 2026-07-22 (research-contaminated: read SPLITS, not levels)",
        "overall": stats_of(base),
        "spot": stats_of(base[~base["is_otc"]]),
        "otc": stats_of(base[base["is_otc"]]),
        "meta_transfer_buckets": {},
        "otc_meta_buckets": {},
        "by_asset_top": {},
    }
    for lo, hi in ((0.0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)):
        key = f"meta_{lo}-{hi}"
        report["meta_transfer_buckets"][key] = stats_of(
            base[(base["meta_p"] >= lo) & (base["meta_p"] < hi)])
        report["otc_meta_buckets"][key] = stats_of(
            base[base["is_otc"] & (base["meta_p"] >= lo) & (base["meta_p"] < hi)])
    counts = base.groupby("asset")["won"].agg(["count", "mean"]).sort_values("count", ascending=False)
    report["by_asset_top"] = {a: f"{r['mean']:.1%} (raw n={int(r['count'])})"
                              for a, r in counts.head(10).iterrows()}

    Path("research_logs/otc_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
