"""Model acceptance report - fixed pass/fail contract, run on demand.

A candidate configuration may advance toward deployment ONLY if every check
passes. Criteria are fixed here, in code, so they cannot drift to fit a
result (structure borrowed from the mantotan/quant-modelling acceptance
contract; statistics from the Lopez de Prado family):

  1. holdout_edge      leak-free holdout win rate beats break-even
                       (one-sided binomial p < 0.0125, the Bonferroni alpha)
  2. pbo               Probability of Backtest Overfitting < 0.40 over the
                       (config x time-block) CSCV matrix
  3. deflated          deflated win-rate p < 0.05, penalised for the HONEST
                       number of attempted experiments (registry.total_trials)
  4. brier             holdout Brier of the direction model < 0.25
  5. ece               meta-model Expected Calibration Error < 0.05 on holdout
  6. volume            >= 200 independent holdout trades
  7. paper             live paper/practice confirmation - ALWAYS reported as
                       PENDING here; only the pre-registered forward test can
                       set it, never this script

Inputs are the research_wr bundles (decade walk-forward predictions). The
report also emits reliability tables (global / pair / fold-year / volatility
regime / meta threshold) and the MinTRL forward-window sizing table.
Research-only; never feeds execution.
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
from sklearn.metrics import brier_score_loss

from registry import total_trials
from research_wr import SPLIT_TS, fit_meta_on_selection, independent, session_of
from train import decide_action
from validation_stats import (
    deflated_win_rate,
    ece,
    min_track_record,
    pbo_cscv,
    reliability_table,
)

PROJECT_DIR = Path(__file__).resolve().parent
BONFERRONI_ALPHA = 0.05 / 4


def build_trades(bundles: list, payout: float):
    allrows = pd.concat([b["preds"] for b in bundles], ignore_index=True)
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
    base["year"] = pd.to_datetime(base["to_ts"], unit="s").dt.year
    base["vol_bucket"] = pd.cut(base["vol_regime"], [0, .33, .67, 1.01],
                                labels=["low", "mid", "high"])
    sel_mask = (base["to_ts"] < SPLIT_TS).to_numpy()
    base["meta_p"] = fit_meta_on_selection(base, sel_mask)  # leak-free
    return base, sel_mask, allrows


def pbo_matrix(base: pd.DataFrame, sel_mask, payout: float, purge_s: int) -> pd.DataFrame:
    """(config x time-block) win-rate matrix over HOLDOUT blocks only.

    Audit correction (H2): the earlier whole-decade matrix included
    selection-era blocks whose meta_p values were the meta model's
    predictions on its own training rows - grossly inflated, which pushed
    PBO toward 0 by construction. Holdout-only blocks are genuinely
    out-of-sample for the meta; the grid now also spans the deployed
    thresholds (0.70/0.775) with a 20-trade cell floor."""
    hold = base[~sel_mask].copy()
    ts = hold["to_ts"].to_numpy()
    edges = np.linspace(ts.min(), ts.max() + 1, 9)
    hold["block"] = np.digitize(hold["to_ts"], edges[1:-1])
    rows = {}
    for ev_m in (0.02, 0.03, 0.04):
        for mt in (0.0, 0.55, 0.60, 0.65, 0.70, 0.775):
            cells = []
            for blk in range(8):
                g = hold[(hold["block"] == blk) & (hold["ev"] > ev_m) & (hold["meta_p"] >= mt)]
                g = independent(g, purge_s)
                cells.append(g["won"].mean() if len(g) >= 20 else np.nan)
            if not any(np.isnan(c) for c in cells):
                rows[f"ev{ev_m}/meta{mt}"] = cells
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=[f"block{b}" for b in range(8)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles", default="research_logs/wr_pooled.pkl",
                    help="glob of research_wr bundles (default: the global model)")
    ap.add_argument("--candidate-ev", type=float, default=0.03)
    ap.add_argument("--candidate-meta", type=float, default=0.65)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--payout", type=float, default=0.87)
    ap.add_argument("--out", default="research_logs/acceptance_report.json")
    args = ap.parse_args()

    purge_s = args.horizon * 60
    breakeven = 1.0 / (1.0 + args.payout)
    bundles = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(args.bundles))]
    base, sel_mask, allrows = build_trades(bundles, args.payout)
    hold = base[~sel_mask]

    cand = independent(
        hold[(hold["ev"] > args.candidate_ev) & (hold["meta_p"] >= args.candidate_meta)],
        purge_s,
    )
    wins, n = int(cand["won"].sum()), len(cand)
    n_trials = total_trials()

    # -- checks 1..6 --
    p_edge = stats.binomtest(wins, n, breakeven, alternative="greater").pvalue if n else 1.0
    pbo = pbo_cscv(pbo_matrix(base, sel_mask, args.payout, purge_s))
    defl = deflated_win_rate(wins, n, breakeven, n_trials)
    # Audit correction (L3): Brier over ALL holdout prediction rows, not
    # the EV-gated tail where the check was near-vacuous.
    hold_all = allrows[allrows["to_ts"] >= SPLIT_TS].dropna(subset=["label_up"])
    brier = float(brier_score_loss(hold_all["label_up"], hold_all["p_up"]))
    hold_gated = hold[hold["ev"] > args.candidate_ev]
    meta_ece = ece(hold_gated["meta_p"].to_numpy(), hold_gated["won"].to_numpy())

    checks = {
        "holdout_edge": {"win_rate": round(wins / n, 4) if n else None, "trades": n,
                         "breakeven": round(breakeven, 4), "p": round(float(p_edge), 6),
                         "pass": bool(n and p_edge < BONFERRONI_ALPHA)},
        "pbo": {**pbo, "pass": bool(pbo["pbo"] < 0.40)},
        "deflated": {**defl, "pass": bool(defl.get("passes_05", False))},
        "brier": {"holdout_brier": round(brier, 5), "pass": bool(brier < 0.25)},
        "ece": {"meta_ece_holdout": round(meta_ece, 4), "pass": bool(meta_ece < 0.05)},
        "volume": {"independent_holdout_trades": n, "pass": bool(n >= 200)},
        "paper": {"status": "PENDING - only the pre-registered forward test "
                            "(forward_eval.py) can set this", "pass": None},
    }
    hard = [k for k, v in checks.items() if v["pass"] is False]
    verdict = ("REJECTED: " + ", ".join(hard)) if hard else \
        "PROVISIONAL PASS - all offline checks pass; paper check pending"

    # -- reliability tables --
    tables = {"global": reliability_table(hold_gated, "meta_p", "won").to_dict("records")}
    for col in ("asset", "year", "vol_bucket"):
        tables[col] = reliability_table(hold_gated, "meta_p", "won", col).to_dict("records")

    # -- MinTRL forward sizing --
    mintrl = {
        f"wr_{int(w * 100)}": min_track_record(w, breakeven, alpha=BONFERRONI_ALPHA)
        for w in (0.58, 0.62, 0.66, 0.70)
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": {"ev_margin": args.candidate_ev, "meta_threshold": args.candidate_meta,
                      "bundles": args.bundles},
        "n_experiments_penalised": n_trials,
        "checks": checks,
        "verdict": verdict,
        "min_track_record_bonferroni": mintrl,
        "reliability": tables,
    }
    (PROJECT_DIR / args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"verdict": verdict,
                      "checks": {k: v["pass"] for k, v in checks.items()},
                      "n_experiments_penalised": n_trials,
                      "min_track_record": mintrl}, indent=2))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
