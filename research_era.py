"""RESEARCH - the 2003-2015 untouched-era holdout (pre-registered).

The registry entry 'era-holdout-2003-2015' was written BEFORE this data was
downloaded: no model was ever trained on these years and no research
decision ever looked at them. The frozen recipe (features v1.3, lgbm
config, meta config, ev 0.03) runs with era-internal walk-forward; the meta
model trains on trades resolved <= 2010 and is scored ONCE on 2011-2015 at
the thresholds FIXED in advance from the 2016-2025 work (0.65/0.70/0.775).

PASS expectation (pre-committed): monotone staircase, win rate above the
53.5% break-even at all three thresholds. Anything else is reported as-is.
"""

import glob
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research_wr import fit_meta_on_selection, independent
from train import decide_action

PAYOUT = 0.87
PURGE_S = 15 * 60
ERA_SPLIT_TS = int(datetime(2011, 1, 1, tzinfo=timezone.utc).timestamp())
FIXED_THRESHOLDS = (0.65, 0.70, 0.775)   # chosen on 2016-2025, frozen here
EV_MARGIN = 0.03
BREAKEVEN = 1.0 / (1.0 + PAYOUT)


def main() -> None:
    bundles = [pickle.load(open(p, "rb"))
               for p in sorted(glob.glob("research_logs/era_*.pkl"))]
    base = pd.concat([b["preds"] for b in bundles], ignore_index=True)
    base["action"] = [decide_action(float(p), PAYOUT, 0.0) for p in base["p_up"]]
    base = base[base["action"] != "no_trade"].reset_index(drop=True)
    base["won"] = ((base["label_up"] == 1.0) == (base["action"] == "binary_call")).astype(float)
    base["is_call"] = (base["action"] == "binary_call").astype(float)
    base["conf"] = (base["p_up"] - 0.5).abs()
    ce = base["p_up"] * PAYOUT - (1 - base["p_up"])
    pe = (1 - base["p_up"]) * PAYOUT - base["p_up"]
    base["ev"] = np.maximum(ce, pe)
    base["ts"] = base["to_ts"].astype(int)

    sel = (base["to_ts"] < ERA_SPLIT_TS).to_numpy()
    base["meta_p"] = fit_meta_on_selection(base, sel)
    hold = base[~sel]

    rows = []
    for thr in FIXED_THRESHOLDS:
        h = independent(hold[(hold["ev"] > EV_MARGIN) & (hold["meta_p"] >= thr)], PURGE_S)
        n = len(h)
        wins = int(h["won"].sum()) if n else 0
        p = (stats.binomtest(wins, n, BREAKEVEN, alternative="greater").pvalue
             if n >= 30 else None)
        rows.append({"meta_threshold": thr, "independent_trades": n,
                     "win_rate": round(wins / n, 4) if n else None,
                     "p_vs_breakeven": float(f"{p:.3e}") if p is not None else None})

    wrs = [r["win_rate"] for r in rows if r["win_rate"] is not None]
    monotone = all(b >= a for a, b in zip(wrs, wrs[1:])) if len(wrs) > 1 else False
    beats = all(w > BREAKEVEN for w in wrs) if wrs else False
    verdict = ("REPRODUCED - staircase monotone and above break-even at all "
               "fixed thresholds") if (monotone and beats) else \
              ("PARTIAL - above break-even but not monotone" if beats else
               "FAILED - below break-even at one or more fixed thresholds")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "era": "2003-2015 (meta selection <= 2010, scored 2011-2015)",
        "breakeven": round(BREAKEVEN, 4),
        "fixed_thresholds": FIXED_THRESHOLDS,
        "results": rows,
        "verdict": verdict,
        "reference_2016_2025": {"0.65": 0.674, "0.70": 0.715, "0.775": 0.791},
    }
    Path("research_logs/era_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
