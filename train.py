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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, FEATURE_VERSION, build_features

MODEL_VERSION = "logreg-1.1.0"  # 1.1.0: chronological purged calibration


def _base_pipeline():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=0)
    )


def _prob_up(model, X) -> np.ndarray:
    classes = list(model.classes_)
    if 1.0 not in classes:
        return np.zeros(len(X))
    if 0.0 not in classes:
        return np.ones(len(X))
    return model.predict_proba(X)[:, classes.index(1.0)]


def chrono_calibration_splits(
    n_rows: int, n_folds: int = 3, gap: int = 5, min_train: int = 50
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological purged calibration folds.

    Rows are assumed time-ordered. The row range is cut into n_folds + 1
    sequential blocks; fold k trains on every row strictly before block k+1
    minus a `gap` purge (overlapping-label protection) and validates on block
    k+1. Every training index therefore precedes every validation index."""
    boundaries = np.linspace(0, n_rows, n_folds + 2, dtype=int)
    splits = []
    for k in range(1, n_folds + 1):
        val_start, val_end = boundaries[k], boundaries[k + 1]
        train_end = max(0, val_start - gap)
        if train_end < min_train or val_end <= val_start:
            continue
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return splits


class ChronoCalibratedModel:
    """Regularized logistic regression with chronologically calibrated output.

    Calibration pairs come only from purged, strictly-later validation blocks
    inside the training window (see chrono_calibration_splits), then a sigmoid
    map (logistic regression on the raw score) is fitted to them. The final
    base model is refit on the whole training window. The test window is
    never touched."""

    def __init__(self, n_folds: int = 3, gap: int = 5):
        self.n_folds = n_folds
        self.gap = gap

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.constant_ = None
        if len(np.unique(y)) < 2:
            # Degenerate training window: emit the only observed class.
            self.constant_ = float(np.asarray(y)[0])
            return self
        scores, labels = [], []
        for train_idx, val_idx in chrono_calibration_splits(len(X), self.n_folds, self.gap):
            base = _base_pipeline()
            base.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(_prob_up(base, X.iloc[val_idx]))
            labels.append(y.iloc[val_idx].to_numpy())

        self.base_ = _base_pipeline()
        self.base_.fit(X, y)

        self.calibrator_ = None
        if scores:
            raw = np.concatenate(scores).reshape(-1, 1)
            out = np.concatenate(labels)
            if len(np.unique(out)) == 2:
                self.calibrator_ = LogisticRegression(max_iter=1000)
                self.calibrator_.fit(raw, out)
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        raw = _prob_up(self.base_, X)
        if self.calibrator_ is None:
            return raw
        return _prob_up(self.calibrator_, raw.reshape(-1, 1))


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
        model = ChronoCalibratedModel(n_folds=3, gap=gap)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        # Model is now frozen; only strictly-later rows are predicted.
        p_up = model.predict_proba_up(X.iloc[test_idx])

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

    from storage import load_canonical_history, open_db

    conn = open_db()
    # Canonical history: ALL immutable datasets merged and deduplicated, so
    # scheduled rolling collections accumulate instead of replacing.
    candles, history_report = load_canonical_history(conn, args.asset, args.interval)
    if candles.empty:
        raise SystemExit(f"no data for {args.asset}@{args.interval}s - run collector.py first")
    if history_report["conflicts"]:
        raise SystemExit(
            f"canonical history has {len(history_report['conflicts'])} candle "
            f"conflicts - resolve before training: {history_report['conflicts'][:3]}"
        )

    feature_frame = build_features(candles, interval=args.interval, horizon=args.horizon)
    result = walk_forward(
        feature_frame,
        payout=args.payout,
        n_splits=args.splits,
        ev_margin=args.ev_margin,
        expiry_seconds=args.horizon * args.interval,
        horizon=args.horizon,
    )
    result["manifest"]["asset"] = args.asset
    result["manifest"]["dataset_content_sha256"] = history_report["content_sha256"]
    result["manifest"]["datasets_used"] = history_report["datasets_used"]
    result["manifest"]["history_gaps"] = len(history_report["gaps"])

    from experiments import count_variants, record_experiment

    entry = record_experiment(
        dataset_content_sha256=history_report["content_sha256"],
        parameters={k: v for k, v in result["manifest"].items() if k != "feature_columns"},
        fold_ranges=[
            {"fold": f["fold"], "train_end_ts": f["train_end_ts"], "test_start_ts": f["test_start_ts"]}
            for f in result["folds"]
        ],
        payout_source="assumed",
        outcome="completed",
    )
    result["manifest"]["experiment_id"] = entry["id"]
    result["manifest"]["variants_attempted"] = count_variants(history_report["content_sha256"])

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "signals.json").write_text(json.dumps(result["signals"], indent=1))
    (out_dir / "manifest.json").write_text(json.dumps(result["manifest"], indent=2))
    (out_dir / "folds.json").write_text(json.dumps(result["folds"], indent=2))

    print(json.dumps({"folds": result["folds"], "manifest": result["manifest"]}, indent=2))


if __name__ == "__main__":
    main()
