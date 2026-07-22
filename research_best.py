"""RESEARCH SCREENING ONLY - find the best deployable option, leak-free.

Consumes the ENRICHED research_wr bundles (p_up + p_up_ext + extra_vol +
extra_mtf columns) and evaluates a fixed candidate family, everything fit on
SELECTION (<= 2022) and scored once on HOLDOUT (>= 2023):

  v1        current meta (production context) at several thresholds
  v2        meta with enriched context: extra_vol + extra_mtf blocks,
            enriched-model probability, base/enriched agreement, and causal
            per-asset streak features (trailing resolved-outcome win rate)
  +cons     consensus gate: base and enriched direction models must agree
  iso       per-pair isotonic recalibration of the meta probability
            (targets the USDJPY overconfidence found by acceptance_report)

Best option = highest holdout win rate subject to a minimum independent
holdout trade count. Also reports the unconstrained maximum for reference.
Registers every variant family in the research registry. Screening only.
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

from features import EXTRA_MTF_COLUMNS, EXTRA_VOL_COLUMNS
from registry import record
from research_wr import META_CONTEXT, SPLIT_TS, independent, session_of
from train import decide_action

PROJECT_DIR = Path(__file__).resolve().parent
HORIZON_S = 15 * 60


def build_base(bundles: list, payout: float) -> tuple[pd.DataFrame, np.ndarray]:
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
    base = base.sort_values(["ts"]).reset_index(drop=True)
    # Consensus and enriched-probability context.
    base["agree"] = (np.sign(base["p_up"] - 0.5) == np.sign(base["p_up_ext"] - 0.5)).astype(float)
    base["conf_ext"] = (base["p_up_ext"] - 0.5).abs()
    # Causal per-asset streaks: trailing win rate over the last N gated
    # signals whose OUTCOME had resolved (to_ts + horizon in the past)
    # before this signal fired. NaN-safe default 0.5.
    for n_lag in (20, 50):
        col = f"streak_{n_lag}"
        vals = np.full(len(base), 0.5)
        for asset, g in base.groupby("asset"):
            idx = g.index.to_numpy()
            ts = g["ts"].to_numpy()
            won = g["won"].to_numpy()
            resolved_upto = np.searchsorted(ts + HORIZON_S, ts, side="right")
            for j, i in enumerate(idx):
                k = resolved_upto[j]
                if k >= 5:
                    lo = max(0, k - n_lag)
                    vals[i] = won[lo:k].mean()
        base[col] = vals
    sel_mask = (base["to_ts"] < SPLIT_TS).to_numpy()
    return base, sel_mask


def fit_meta(base: pd.DataFrame, sel_mask: np.ndarray, feat: list[str],
             isotonic_by_pair: bool = False) -> np.ndarray:
    from lightgbm import LGBMClassifier

    feat = [c for c in feat if c in base.columns]
    for pair in base["asset"].str.replace("-OTC", "", regex=False).unique():
        col = f"pair_{pair}"
        base[col] = (base["asset"].str.replace("-OTC", "", regex=False) == pair).astype(float)
        if col not in feat:
            feat.append(col)
    meta = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, verbosity=-1,
    )
    meta.fit(base.loc[sel_mask, feat], base.loc[sel_mask, "won"])
    p = meta.predict_proba(base[feat])[:, 1]
    if isotonic_by_pair:
        from sklearn.isotonic import IsotonicRegression

        p = p.copy()
        for asset in base["asset"].unique():
            m = (base["asset"] == asset).to_numpy()
            m_sel = m & sel_mask
            if m_sel.sum() < 500:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p[m_sel], base.loc[m_sel, "won"])
            p[m] = iso.predict(p[m])
    return p


def score(base: pd.DataFrame, sel_mask: np.ndarray, mask_extra, meta_col: str,
          thresholds, ev_margin: float, breakeven: float) -> list[dict]:
    out = []
    hold = base[~sel_mask]
    for thr in thresholds:
        m = (hold["ev"] > ev_margin) & (hold[meta_col] >= thr)
        if mask_extra is not None:
            m &= mask_extra(hold)
        h = independent(hold[m], HORIZON_S)
        n = len(h)
        if n < 30:
            out.append({"threshold": thr, "trades": n, "note": "too thin"})
            continue
        wins = int(h["won"].sum())
        p = stats.binomtest(wins, n, breakeven, alternative="greater").pvalue
        out.append({"threshold": thr, "trades": n,
                    "wr": round(wins / n, 4), "p": float(f"{p:.3e}")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles", default="research_logs/wre_*.pkl")
    ap.add_argument("--payout", type=float, default=0.87)
    ap.add_argument("--ev-margin", type=float, default=0.03)
    ap.add_argument("--min-trades", type=int, default=200)
    ap.add_argument("--out", default="research_logs/best_option.json")
    args = ap.parse_args()

    breakeven = 1.0 / (1.0 + args.payout)
    bundles = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(args.bundles))]
    base, sel = build_base(bundles, args.payout)
    thresholds = (0.60, 0.65, 0.70, 0.75, 0.775)

    v2_context = (META_CONTEXT + EXTRA_VOL_COLUMNS + EXTRA_MTF_COLUMNS
                  + ["p_up_ext", "conf_ext", "agree", "streak_20", "streak_50"])

    base["meta_v1"] = fit_meta(base, sel, META_CONTEXT)
    base["meta_v2"] = fit_meta(base, sel, v2_context)
    base["meta_v2_iso"] = fit_meta(base, sel, v2_context, isotonic_by_pair=True)

    consensus = lambda h: h["agree"] == 1.0  # noqa: E731
    candidates = {
        "v1": score(base, sel, None, "meta_v1", thresholds, args.ev_margin, breakeven),
        "v1+cons": score(base, sel, consensus, "meta_v1", thresholds, args.ev_margin, breakeven),
        "v2": score(base, sel, None, "meta_v2", thresholds, args.ev_margin, breakeven),
        "v2+cons": score(base, sel, consensus, "meta_v2", thresholds, args.ev_margin, breakeven),
        "v2_iso": score(base, sel, None, "meta_v2_iso", thresholds, args.ev_margin, breakeven),
        "v2_iso+cons": score(base, sel, consensus, "meta_v2_iso", thresholds, args.ev_margin, breakeven),
    }

    flat = [{"family": fam, **row} for fam, rows in candidates.items()
            for row in rows if "wr" in row]
    eligible = [r for r in flat if r["trades"] >= args.min_trades]
    best = max(eligible, key=lambda r: r["wr"]) if eligible else None
    best_any = max(flat, key=lambda r: r["wr"]) if flat else None

    n_variants = sum(len(rows) for rows in candidates.values())
    record("best-option-sweep",
           "meta v1/v2/iso x consensus x thresholds on enriched decade bundles",
           n_variants,
           config={"ev_margin": args.ev_margin, "thresholds": thresholds},
           outcome=f"best@min{args.min_trades}: {best}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundles": args.bundles, "breakeven": round(breakeven, 4),
        "candidates": candidates,
        "best_with_min_trades": best,
        "best_unconstrained": best_any,
    }
    (PROJECT_DIR / args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["candidates"], indent=2))
    print("\nBEST (>= %d trades): %s" % (args.min_trades, json.dumps(best)))
    print("BEST (unconstrained): %s" % json.dumps(best_any))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
