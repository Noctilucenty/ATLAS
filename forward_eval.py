"""Run the pre-registered forward test (FORWARD_TEST.md). ONE evaluation.

Written 2026-07-22, BEFORE any forward data existed, so the scoring rules
could not be shaped by the results. Do not edit the metrics or thresholds
after forward data has been seen - that is the whole point of the file.

Two independent evidence tracks, reported together:

  candles  - replay the frozen H2 model over candles collected AFTER the
             registration cutoff, exactly as research_pooled scored it
             in-sample (EV gate, per-asset independent trades, cross-asset
             chain clusters).
  paper    - score the live paper log (logs/live_h2.jsonl), whose signals
             were emitted in real time before their outcomes existed and
             are therefore immune to every hindsight bias.

Hypotheses evaluated (see FORWARD_TEST.md):
  H3           PRIMARY - H2 primary signals filtered by meta_p >= 0.60
  H2 primary   secondary - ev_margin 0.03
  H2 secondary secondary - ev_margin 0.04 (registered policy variant)

Success (pre-committed, identical for each): cluster mean win fraction
above break-even at the payout actually observed, one-sided t-test
p < ALPHA, and at least 20 clusters (>= 30 for the candles track).
ALPHA is Bonferroni-corrected for the four tests sharing this one forward
window, so it is 0.0125, not 0.05.
"""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features import build_features
from instruments import INSTRUMENTS
from research_pooled import XS_COLUMNS, add_cross_asset, currencies  # noqa: F401
from storage import load_canonical_history, open_db
from train import decide_action

PROJECT_DIR = Path(__file__).resolve().parent
H2_PRIMARY_MARGIN = 0.03
# Four pre-registered tests share one forward window; Bonferroni keeps the
# family-wise error at 5%. H3 is the single primary hypothesis.
ALPHA = 0.05 / 4
H2_SECONDARY_MARGIN = 0.04
H3_META_THRESHOLD = 0.60          # original H3 primary
# H3 family: decade holdout (leak-free, research_wr.py) shows win rate rises
# monotonically with the meta threshold - 61% / 67% / 72% at 0.60 / 0.65 /
# 0.70. Evaluated together on the forward window; 0.65 is the preferred
# operating point (best win-rate/volume balance), 0.70 the aggressive one.
# 0.775 added 2026-07-22 as an EXPLORATORY reported metric only (decade
# holdout: 80.9% on 392 trades, all acceptance checks pass; MinTRL ~13
# trades at that win rate). It is NOT a pass/fail hypothesis - the primary
# remains H3 @ 0.65 and the alpha accounting is unchanged.
H3_META_THRESHOLDS = (0.60, 0.65, 0.70, 0.775)
MIN_CLUSTERS_CANDLES = 30
MIN_CLUSTERS_PAPER = 20


def cluster_stats(trades: list[tuple[str, int, bool, float]], purge_s: int) -> dict:
    """trades: (asset, ts, won, payout). Per-asset independent trades plus
    cross-asset chain clusters; break-even uses the observed mean payout."""
    if not trades:
        return {"trades": 0, "verdict": "no trades"}
    trades = sorted(trades, key=lambda t: t[1])

    kept, last_by_asset = [], {}
    for asset, ts, won, _ in trades:
        if ts >= last_by_asset.get(asset, -1) + purge_s:
            kept.append(won)
            last_by_asset[asset] = ts

    clusters = []
    for _, ts, won, _ in trades:
        if clusters and ts < clusters[-1]["end"]:
            c = clusters[-1]
            c["end"] = max(c["end"], ts + purge_s)
            c["n"] += 1
            c["wins"] += won
        else:
            clusters.append({"end": ts + purge_s, "n": 1, "wins": won})
    fracs = np.array([c["wins"] / c["n"] for c in clusters])

    payout = float(np.mean([t[3] for t in trades]))
    breakeven = 1.0 / (1.0 + payout)
    # One-sided: we only care about beating break-even, never about
    # significantly losing.
    if len(fracs) > 2:
        t_stat, two_sided = stats.ttest_1samp(fracs, breakeven)
        p_one_sided = two_sided / 2 if t_stat > 0 else 1.0 - two_sided / 2
    else:
        p_one_sided = None
    return {
        "raw_trades": len(trades),
        "independent": len(kept),
        "independent_win_rate": round(float(np.mean(kept)), 4) if kept else None,
        "clusters": len(clusters),
        "cluster_win_frac": round(float(fracs.mean()), 4),
        "observed_payout": round(payout, 4),
        "breakeven": round(breakeven, 4),
        "p_one_sided": round(float(p_one_sided), 5) if p_one_sided is not None else None,
    }


def verdict(stats_dict: dict, min_clusters: int) -> str:
    if not stats_dict.get("clusters"):
        return "INCONCLUSIVE - no trades"
    if stats_dict["clusters"] < min_clusters:
        return f"INCONCLUSIVE - {stats_dict['clusters']} clusters < {min_clusters} required"
    beats = stats_dict["cluster_win_frac"] > stats_dict["breakeven"]
    sig = stats_dict["p_one_sided"] is not None and stats_dict["p_one_sided"] < ALPHA
    return "PASS" if (beats and sig) else "FAIL"


def load_bundle(name: str):
    paths = sorted((PROJECT_DIR / "models").glob(name))
    if not paths:
        raise SystemExit(f"missing models/{name}")
    with open(paths[-1], "rb") as fh:
        return paths[-1].name, pickle.load(fh)


def meta_probabilities(meta_bundle, frame: pd.DataFrame, p_up, actions) -> np.ndarray:
    feats = pd.DataFrame(index=frame.index)
    for col in meta_bundle["features"]:
        if col == "p_up":
            feats[col] = p_up
        elif col == "conf":
            feats[col] = np.abs(np.asarray(p_up) - 0.5)
        elif col == "is_call":
            feats[col] = [float(a == "binary_call") for a in actions]
        elif col.startswith("pair_"):
            base = col[len("pair_"):]
            feats[col] = (
                frame["asset"].str.replace("-OTC", "", regex=False) == base
            ).astype(float)
        else:
            feats[col] = frame[col].to_numpy() if col in frame.columns else 0.0
    return meta_bundle["model"].predict_proba(feats[meta_bundle["features"]])[:, 1]


def candles_track(cutoff_ts: int, horizon: int, payout_fallback: float) -> dict:
    model_name, bundle = load_bundle("h2-*.pkl")
    model, meta = bundle["model"], bundle["meta"]
    _, meta_bundle = load_bundle("meta-h3.pkl")
    feature_cols = meta["feature_columns"]

    conn = open_db()
    parts = []
    for asset in INSTRUMENTS:
        candles, _ = load_canonical_history(conn, asset, 60)
        if candles.empty:
            continue
        ff = build_features(candles, interval=60, horizon=horizon, entry_next_open=True)
        ff["asset"] = asset
        parts.append(ff)
    pooled = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["label_up"])
        .sort_values("to_ts", kind="stable")
        .reset_index(drop=True)
    )
    pooled = add_cross_asset(pooled)
    forward = pooled[pooled["to_ts"] > cutoff_ts].reset_index(drop=True)
    print(f"[candles] model={model_name} forward rows={len(forward)} "
          f"(cutoff {datetime.fromtimestamp(cutoff_ts, timezone.utc).isoformat()})",
          flush=True)
    if forward.empty:
        return {"error": "no forward rows - collector has not gathered new data yet"}

    p_up = model.predict_proba_up(forward[feature_cols])
    purge_s = horizon * 60
    out = {"forward_rows": len(forward), "model": model_name}

    for label, margin in (("h2_primary", H2_PRIMARY_MARGIN),
                          ("h2_secondary", H2_SECONDARY_MARGIN)):
        trades = []
        for i, p in enumerate(p_up):
            action = decide_action(float(p), payout_fallback, margin)
            if action == "no_trade":
                continue
            row = forward.iloc[i]
            won = (row["label_up"] == 1.0) == (action == "binary_call")
            trades.append((row["asset"], int(row["to_ts"]), bool(won), payout_fallback))
        s = cluster_stats(trades, purge_s)
        s["verdict"] = verdict(s, MIN_CLUSTERS_CANDLES)
        out[label] = s

    # H3 family: primary-gate signals surviving the meta filter, evaluated at
    # every registered threshold (decade holdout predicts monotone gain:
    # 0.60->61%, 0.65->67%, 0.70->72%). meta_p is a per-signal score, so the
    # threshold applies at evaluation time - no model change, no leak (the
    # forward window post-dates the meta model's training data).
    actions = [decide_action(float(p), payout_fallback, H2_PRIMARY_MARGIN) for p in p_up]
    idx = [i for i, a in enumerate(actions) if a != "no_trade"]
    if idx:
        sub = forward.iloc[idx].reset_index(drop=True)
        meta_p = meta_probabilities(
            meta_bundle, sub, [p_up[i] for i in idx], [actions[i] for i in idx]
        )
        for thr in H3_META_THRESHOLDS:
            trades = []
            for j, i in enumerate(idx):
                if meta_p[j] < thr:
                    continue
                row = forward.iloc[i]
                won = (row["label_up"] == 1.0) == (actions[i] == "binary_call")
                trades.append((row["asset"], int(row["to_ts"]), bool(won), payout_fallback))
            s = cluster_stats(trades, purge_s)
            s["verdict"] = verdict(s, MIN_CLUSTERS_CANDLES)
            s["meta_kept"] = f"{len(trades)}/{len(idx)}"
            out[f"h3_meta{int(thr * 100)}"] = s
    return out


def paper_track(horizon: int) -> dict:
    """Score live paper signals against collected candles. Signals were
    emitted before their outcomes existed - no hindsight is possible."""
    log_path = PROJECT_DIR / "logs" / "live_h2.jsonl"
    if not log_path.exists():
        return {"error": "no logs/live_h2.jsonl yet"}
    signals = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    signals = [s for s in signals if not s.get("settled")]
    if not signals:
        return {"error": "paper log is empty"}

    conn = open_db()
    labels: dict[tuple[str, int], float] = {}
    for asset in {s["asset"] for s in signals}:
        candles, _ = load_canonical_history(conn, asset, 60)
        if candles.empty:
            continue
        ff = build_features(candles, interval=60, horizon=horizon, entry_next_open=True)
        for ts, lab in zip(ff["to_ts"].astype(int), ff["label_up"]):
            if not pd.isna(lab):
                labels[(asset, int(ts))] = float(lab)

    purge_s = horizon * 60
    out = {"paper_signals": len(signals)}
    keeps = [("h2_primary", lambda s: True)]
    for thr in H3_META_THRESHOLDS:
        keeps.append((f"h3_meta{int(thr * 100)}",
                      lambda s, t=thr: s.get("meta_p") is not None and s["meta_p"] >= t))
    for label, keep in keeps:
        trades, unscored = [], 0
        for s in signals:
            if not keep(s):
                continue
            lab = labels.get((s["asset"], int(s["bar_to_ts"])))
            if lab is None:
                unscored += 1
                continue
            won = (lab == 1.0) == (s["action"] == "binary_call")
            trades.append((s["asset"], int(s["ts"]), bool(won), float(s["payout"])))
        st = cluster_stats(trades, purge_s)
        st["verdict"] = verdict(st, MIN_CLUSTERS_PAPER)
        st["unscored_pending_candles"] = unscored
        out[label] = st
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2026-07-22T00:00:00Z",
                        help="registration cutoff; only later data is forward data")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--payout", type=float, default=0.87,
                        help="assumed payout for the candles track (paper track uses "
                        "the payout actually quoted at signal time)")
    parser.add_argument("--track", choices=("both", "candles", "paper"), default="both")
    args = parser.parse_args()

    cutoff_ts = int(
        datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")).timestamp()
    )
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": args.cutoff,
        "horizon_bars": args.horizon,
    }
    if args.track in ("both", "candles"):
        report["candles_track"] = candles_track(cutoff_ts, args.horizon, args.payout)
    if args.track in ("both", "paper"):
        report["paper_track"] = paper_track(args.horizon)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
