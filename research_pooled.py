"""RESEARCH SCREENING ONLY - pooled cross-asset walk-forward.

Trains one model on all registered instruments at once (features are
asset-agnostic) to multiply effective sample size, and evaluates with
TIME-based purged folds: with ~10 assets interleaved per minute, row-based
gaps under-purge, so train/test boundaries are cut on to_ts and training
rows whose label window crosses the test start are dropped.

Options over the frozen FORWARD_TEST.md config (hypothesis #1):
  --cross-asset      add causal currency-strength basket features computed
                     across the whole pool (hypothesis #2)
  --entry-next-open  realistic-execution labels: strike = next bar's open
  --ev-margins       evaluate several EV gates from one set of predictions

Every evaluation reports per-asset independent trades AND cross-asset chain
clusters (trades whose label windows overlap are one cluster = one bet).

Prints metrics; writes NO run bundle and must never feed the execution
path - promote a promising config through train.py first.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss

from features import EXTRA_VOL_COLUMNS, FEATURE_COLUMNS, build_features
from instruments import INSTRUMENTS
from storage import load_canonical_history, open_db
from train import ChronoCalibratedModel, decide_action

XS_COLUMNS = ["xs_base_str", "xs_quote_str", "xs_mkt_vol"]


def currencies(asset: str) -> tuple[str, str]:
    pair = asset.replace("-OTC", "")
    return pair[:3], pair[3:6]


def load_pooled(
    conn,
    interval: int,
    horizon: int,
    entry_next_open: bool,
    extra_vol: bool = False,
) -> pd.DataFrame:
    parts = []
    for asset in INSTRUMENTS:
        candles, _ = load_canonical_history(conn, asset, interval)
        if candles.empty:
            continue
        ff = build_features(
            candles, interval=interval, horizon=horizon,
            entry_next_open=entry_next_open, extra_vol=extra_vol,
        )
        ff["asset"] = asset
        parts.append(ff)
    pooled = pd.concat(parts, ignore_index=True).dropna(subset=["label_up"])
    return pooled.sort_values("to_ts", kind="stable").reset_index(drop=True)


def add_cross_asset(pooled: pd.DataFrame) -> pd.DataFrame:
    """Causal currency-strength features from the SAME timestamp's ret_5
    across the pool (each pair's own contribution excluded).

    xs_base_str / xs_quote_str: mean signed ret_5 of the OTHER pairs sharing
    this pair's base/quote currency (+ret if the currency is base there,
    -ret if quote). xs_mkt_vol: cross-sectional mean |ret_5| (activity)."""
    wide = pooled.pivot_table(index="to_ts", columns="asset", values="ret_5")
    strength_num = {}
    strength_cnt = {}
    for asset in wide.columns:
        base, quote = currencies(asset)
        col = wide[asset]
        for cur, sign in ((base, 1.0), (quote, -1.0)):
            signed = col * sign
            strength_num[cur] = strength_num.get(cur, 0.0) + signed.fillna(0.0)
            strength_cnt[cur] = strength_cnt.get(cur, 0) + signed.notna().astype(int)
    mkt_vol = wide.abs().mean(axis=1)

    idx = pooled["to_ts"]
    base_str = np.empty(len(pooled))
    quote_str = np.empty(len(pooled))
    for asset in wide.columns:
        mask = (pooled["asset"] == asset).to_numpy()
        ts = idx[mask]
        base, quote = currencies(asset)
        own = wide[asset].reindex(ts).fillna(0.0).to_numpy()
        own_n = wide[asset].reindex(ts).notna().astype(int).to_numpy()
        for cur, sign, dest in ((base, 1.0, base_str), (quote, -1.0, quote_str)):
            num = strength_num[cur].reindex(ts).to_numpy() - own * sign
            cnt = strength_cnt[cur].reindex(ts).to_numpy() - own_n
            with np.errstate(invalid="ignore"):
                dest[mask] = np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0)
    pooled = pooled.copy()
    pooled["xs_base_str"] = base_str
    pooled["xs_quote_str"] = quote_str
    pooled["xs_mkt_vol"] = mkt_vol.reindex(idx).fillna(0.0).to_numpy()
    return pooled


def time_folds(pooled: pd.DataFrame, n_splits: int, purge_s: int):
    """Equal time blocks; fold k tests block k+1, trains on everything whose
    label window ends before the test block starts (time purge)."""
    ts = pooled["to_ts"].to_numpy()
    edges = np.linspace(ts[0], ts[-1], n_splits + 2)
    for k in range(1, n_splits + 1):
        test_mask = (ts > edges[k]) & (ts <= edges[k + 1])
        train_mask = ts <= edges[k] - purge_s
        if test_mask.sum() and train_mask.sum() > 1000:
            yield np.where(train_mask)[0], np.where(test_mask)[0]


def evaluate_margin(preds: list, margin: float, payout: float, purge_s: int) -> dict:
    breakeven = 1.0 / (1.0 + payout)
    trades = []
    for asset, ts, p, label in preds:
        action = decide_action(p, payout, margin)
        if action == "no_trade":
            continue
        won = (label == 1.0) == (action == "binary_call")
        trades.append((asset, ts, bool(won)))
    trades.sort(key=lambda t: t[1])

    # Per-asset independent trades.
    kept, last_by_asset = [], {}
    for asset, ts, won in trades:
        if ts >= last_by_asset.get(asset, -1) + purge_s:
            kept.append((asset, won))
            last_by_asset[asset] = ts
    wins, n = sum(w for _, w in kept), len(kept)

    # Cross-asset chain clusters: one overlapping burst = one bet.
    clusters = []
    for asset, ts, won in trades:
        if clusters and ts < clusters[-1]["end"]:
            c = clusters[-1]
            c["end"] = max(c["end"], ts + purge_s)
            c["n"] += 1
            c["wins"] += won
        else:
            clusters.append({"end": ts + purge_s, "n": 1, "wins": won})
    fracs = np.array([c["wins"] / c["n"] for c in clusters]) if clusters else np.array([])
    out = {
        "ev_margin": margin,
        "raw_trades": len(trades),
        "independent": n,
        "win_rate": round(wins / n, 4) if n else None,
        # One-sided vs the economic break-even, not a coin flip (audit M1) -
        # "beats 0.5" is not the claim that matters at a 0.87 payout.
        "p_ind": (round(stats.binomtest(wins, n, breakeven,
                                        alternative="greater").pvalue, 6)
                  if n else None),
        "clusters": len(clusters),
        "cluster_win_frac": round(float(fracs.mean()), 4) if len(fracs) else None,
        "p_cluster": (
            round(float(stats.ttest_1samp(fracs, breakeven).pvalue / 2), 6)
            if len(fracs) > 2 and float(np.std(fracs)) > 0
            and float(np.mean(fracs)) > breakeven
            else None
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--payout", type=float, default=0.87)
    parser.add_argument("--ev-margins", default="0.02",
                        help="comma-separated EV gates evaluated from one prediction pass")
    parser.add_argument("--cross-asset", action="store_true",
                        help="add currency-strength basket features")
    parser.add_argument("--entry-next-open", action="store_true",
                        help="realistic-execution labels (strike = next bar open)")
    parser.add_argument("--extra-vol", action="store_true",
                        help="hypothesis #4: add the range-volatility feature block")
    parser.add_argument("--dump-trades", default=None,
                        help="write (asset, ts, p_up, label) predictions to this JSON path")
    args = parser.parse_args()

    purge_s = args.horizon * args.interval
    pooled = load_pooled(
        open_db(), args.interval, args.horizon, args.entry_next_open,
        extra_vol=args.extra_vol,
    )
    feature_cols = list(FEATURE_COLUMNS)
    if args.cross_asset:
        pooled = add_cross_asset(pooled)
        feature_cols += XS_COLUMNS
    if args.extra_vol:
        feature_cols += EXTRA_VOL_COLUMNS
    print(f"pooled rows={len(pooled)} assets={pooled['asset'].nunique()} "
          f"features={len(feature_cols)} entry_next_open={args.entry_next_open}", flush=True)

    X_all = pooled[feature_cols]
    y_all = pooled["label_up"]
    n_assets = pooled["asset"].nunique()
    folds, preds = [], []
    for fold, (tr, te) in enumerate(time_folds(pooled, args.splits, purge_s)):
        model = ChronoCalibratedModel(
            n_folds=3, gap=args.horizon * n_assets, model_kind="lgbm"
        )
        model.fit(X_all.iloc[tr], y_all.iloc[tr])
        p_up = model.predict_proba_up(X_all.iloc[te])
        brier = float(brier_score_loss(y_all.iloc[te].to_numpy(), p_up))
        folds.append(brier)
        rows = pooled.iloc[te]
        preds.extend(
            zip(rows["asset"], rows["to_ts"].astype(int), map(float, p_up), rows["label_up"])
        )
        print(f"fold {fold}: brier={brier:.5f} test_rows={len(te)}", flush=True)

    print("mean brier:", round(float(np.mean(folds)), 5))
    if args.dump_trades:
        with open(args.dump_trades, "w") as fh:
            json.dump(preds, fh)
    results = [
        evaluate_margin(preds, float(m), args.payout, purge_s)
        for m in args.ev_margins.split(",")
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
