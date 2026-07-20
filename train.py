"""Walk-forward train-freeze-predict orchestrator.

For each walk-forward fold: fit a model ONLY on the training window, freeze
it, predict the strictly later test window, and emit timestamped signals in
MIDAS BinarySignal JSON form. sklearn's TimeSeriesSplit keeps test windows
after training windows; a `gap` of at least the label horizon prevents
overlapping-label leakage across the boundary.

The decision policy is deterministic expected value, not the model:
  call_ev = p_up * payout - (1 - p_up)
  put_ev  = (1 - p_up) * payout - p_up
A trade is emitted only when EV exceeds `ev_margin`; otherwise `no_trade`.
Historical payouts are not reconstructable, so the payout used is ASSUMED and
recorded in the manifest - results must be read as conditional on it.

Model provenance: every signal carries model_version and a feature hash so
backtest results trace to the exact code that produced them.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, FEATURE_VERSION, build_features

MODEL_VERSION = "logreg-1.0.0"


def feature_hash() -> str:
    payload = FEATURE_VERSION + "|" + ",".join(FEATURE_COLUMNS)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def decide_action(p_up: float, payout: float, ev_margin: float) -> str:
    """Deterministic EV policy. Returns binary_call / binary_put / no_trade."""
    call_ev = p_up * payout - (1.0 - p_up)
    put_ev = (1.0 - p_up) * payout - p_up
    if call_ev > ev_margin and call_ev >= put_ev:
        return "binary_call"
    if put_ev > ev_margin:
        return "binary_put"
    return "no_trade"


def _iso_utc(epoch: int) -> str:
    return (
        datetime.fromtimestamp(int(epoch), timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def make_signal(
    to_ts: int,
    p_up: float,
    payout: float,
    ev_margin: float,
    stake: float,
    expiry_seconds: int,
    fold: int,
) -> dict:
    """One MIDAS-compatible BinarySignal JSON object (serde field names)."""
    return {
        "timestamp": _iso_utc(to_ts),
        "action": decide_action(p_up, payout, ev_margin),
        "stake": stake,
        "expiry_seconds": expiry_seconds,
        "payout": payout,
        "predicted_prob_up": round(float(p_up), 6),
        "model_version": MODEL_VERSION,
        "feature_hash": feature_hash(),
        "note": f"fold={fold}",
    }


def walk_forward(
    feature_frame: pd.DataFrame,
    payout: float,
    n_splits: int = 5,
    gap: int | None = None,
    ev_margin: float = 0.02,
    stake: float = 1.0,
    expiry_seconds: int = 300,
    horizon: int = 5,
) -> dict:
    """Run the train-freeze-predict loop. Returns signals + fold metrics."""
    labeled = feature_frame.dropna(subset=["label_up"]).reset_index(drop=True)
    X = labeled[FEATURE_COLUMNS]
    y = labeled["label_up"]
    gap = horizon if gap is None else gap

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    signals: list[dict] = []
    folds: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        # Regularized logistic regression wrapped in cross-validated sigmoid
        # calibration: when out-of-fold predictions carry no information the
        # calibrated probabilities collapse to the base rate, so the EV gate
        # abstains on noise instead of trading overconfident estimates. The
        # calibration folds are contiguous blocks inside the training window
        # only - the test window stays untouched.
        model = CalibratedClassifierCV(
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1000, random_state=0),
            ),
            method="sigmoid",
            cv=3,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        # Model is now frozen; only strictly-later rows are predicted.
        up_col = list(model.classes_).index(1.0)
        p_up = model.predict_proba(X.iloc[test_idx])[:, up_col]

        fold_signals = [
            make_signal(ts, p, payout, ev_margin, stake, expiry_seconds, fold)
            for ts, p in zip(labeled["to_ts"].iloc[test_idx], p_up)
        ]
        signals.extend(fold_signals)

        y_test = y.iloc[test_idx].to_numpy()
        trades = sum(1 for s in fold_signals if s["action"] != "no_trade")
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_end_ts": int(labeled["to_ts"].iloc[train_idx].max()),
                "test_start_ts": int(labeled["to_ts"].iloc[test_idx].min()),
                "brier": float(brier_score_loss(y_test, p_up)),
                "base_rate_up": float(y_test.mean()),
                "trades": trades,
                "coverage": trades / len(test_idx),
            }
        )

    return {
        "signals": signals,
        "folds": folds,
        "manifest": {
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_hash": feature_hash(),
            "feature_columns": FEATURE_COLUMNS,
            "assumed_payout": payout,
            "payout_source": "assumed",  # historical payouts not reconstructable
            "ev_margin": ev_margin,
            "stake": stake,
            "expiry_seconds": expiry_seconds,
            "horizon_bars": horizon,
            "n_splits": n_splits,
            "gap_bars": gap,
            "labeled_rows": len(labeled),
            # Honest search accounting (backtest-overfitting exposure): counts
            # every model configuration tried during development, not just the
            # survivor. v1 history: plain logreg, C=0.1 logreg, calibrated C=0.1.
            "variants_attempted": 3,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="EURUSD")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5, help="label horizon in bars")
    parser.add_argument("--payout", type=float, default=0.85, help="ASSUMED payout ratio")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--ev-margin", type=float, default=0.02)
    parser.add_argument("--out", default="signals_out")
    args = parser.parse_args()

    from storage import latest_dataset_id, load_candles, open_db

    conn = open_db()
    dataset_id = latest_dataset_id(conn, args.asset, args.interval)
    if dataset_id is None:
        raise SystemExit(f"no dataset for {args.asset}@{args.interval}s - run collector.py first")
    candles = load_candles(conn, dataset_id)

    feature_frame = build_features(candles, interval=args.interval, horizon=args.horizon)
    result = walk_forward(
        feature_frame,
        payout=args.payout,
        n_splits=args.splits,
        ev_margin=args.ev_margin,
        expiry_seconds=args.horizon * args.interval,
        horizon=args.horizon,
    )
    result["manifest"]["dataset_id"] = dataset_id
    result["manifest"]["asset"] = args.asset

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "signals.json").write_text(json.dumps(result["signals"], indent=1))
    (out_dir / "manifest.json").write_text(json.dumps(result["manifest"], indent=2))
    (out_dir / "folds.json").write_text(json.dumps(result["folds"], indent=2))

    print(json.dumps({"folds": result["folds"], "manifest": result["manifest"]}, indent=2))


if __name__ == "__main__":
    main()
