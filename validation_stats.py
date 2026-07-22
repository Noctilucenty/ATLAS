"""Overfitting-aware validation statistics (López de Prado family).

Implements the four checks ATLAS lacked, adapted from portfolio-Sharpe form
to the Bernoulli win/lose world of fixed-expiry binaries:

  PBO      Probability of Backtest Overfitting via CSCV (Bailey et al.):
           split the fold axis of a (config x fold) performance matrix into
           all C(S, S/2) in-sample/out-of-sample combinations; PBO is the
           fraction where the IS-best config ranks below the OOS median.
  DEFLATED WIN RATE  The binary-outcome analogue of the Deflated Sharpe
           Ratio: the observed win-rate z-score against break-even is judged
           against the expected maximum of N standard normals, where N is
           the HONEST number of experiments attempted (from the registry) -
           not just the ones we liked.
  MinTRL   Minimum Track Record Length: how many independent trades the
           forward window must contain before an observed win rate can be
           distinguished from break-even at a given alpha.
  ECE      Expected Calibration Error + reliability tables for any
           probability column (meta_p, p_up) over any grouping.

Pure functions over dataframes/arrays; no I/O, no model fitting. Used by
research code and the acceptance report - never by the execution path.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- PBO (CSCV)

def pbo_cscv(perf: pd.DataFrame) -> dict:
    """perf: rows = configs, columns = time folds, values = the metric
    (e.g. fold win rate). Uses all C(S, S/2) fold combinations; S must be
    even and >= 4. Returns PBO plus the logit distribution summary."""
    S = perf.shape[1]
    if S < 4 or S % 2:
        raise ValueError(f"need an even number >= 4 of folds, got {S}")
    cols = list(perf.columns)
    logits = []
    for is_cols in combinations(cols, S // 2):
        oos_cols = [c for c in cols if c not in is_cols]
        is_perf = perf[list(is_cols)].mean(axis=1)
        oos_perf = perf[oos_cols].mean(axis=1)
        best = is_perf.idxmax()
        # Relative OOS rank of the IS-chosen config, in (0, 1).
        rank = (oos_perf < oos_perf[best]).mean() + 0.5 * (oos_perf == oos_perf[best]).mean()
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {
        "pbo": round(float((logits < 0).mean()), 4),
        "n_combinations": len(logits),
        "n_configs": perf.shape[0],
        "median_logit": round(float(np.median(logits)), 4),
    }


# ------------------------------------------------- deflated win rate (DSR analogue)

def expected_max_z(n_trials: int) -> float:
    """Expected maximum of n_trials iid standard normals (Bailey/Prado
    approximation). The bar any 'best result' must clear when it was
    selected from n_trials attempts."""
    if n_trials <= 1:
        return 0.0
    emc = 0.5772156649015329  # Euler-Mascheroni
    a = stats.norm.ppf(1 - 1.0 / n_trials)
    b = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float((1 - emc) * a + emc * b)


def deflated_win_rate(wins: int, n: int, breakeven: float, n_trials: int) -> dict:
    """Is an observed win rate significant AFTER accounting for having tried
    n_trials experiment variants? z is the binomial z-score against
    break-even; the deflated p-value asks P(Z - E[max of n_trials] > z)."""
    if n == 0:
        return {"error": "no trades"}
    wr = wins / n
    se = np.sqrt(breakeven * (1 - breakeven) / n)
    z = (wr - breakeven) / se
    z_bar = expected_max_z(n_trials)
    p_deflated = float(1 - stats.norm.cdf(z - z_bar))
    return {
        "win_rate": round(wr, 4),
        "n_trades": n,
        "z_raw": round(float(z), 3),
        "n_trials_penalised_for": n_trials,
        "expected_max_z_under_null": round(z_bar, 3),
        "z_deflated": round(float(z - z_bar), 3),
        "p_deflated": round(p_deflated, 5),
        "passes_05": bool(p_deflated < 0.05),
    }


# ------------------------------------------------------------------- MinTRL

def min_track_record(wr: float, breakeven: float, alpha: float = 0.05) -> int:
    """Independent trades needed before a true win rate `wr` is
    distinguishable from break-even at one-sided alpha."""
    if wr <= breakeven:
        return -1  # never
    z = stats.norm.ppf(1 - alpha)
    n = (z * np.sqrt(wr * (1 - wr)) / (wr - breakeven)) ** 2
    return int(np.ceil(n))


# ------------------------------------------------------------- calibration

def ece(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error: bin-weighted |mean prob - mean outcome|."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi)
        if m.sum():
            total += m.mean() * abs(prob[m].mean() - outcome[m].mean())
    return float(total)


def reliability_table(df: pd.DataFrame, prob_col: str, outcome_col: str,
                      group_col: str | None = None, bins: int = 5) -> pd.DataFrame:
    """Reliability rows (predicted vs realised) overall or per group."""
    def one(g: pd.DataFrame) -> dict:
        return {
            "n": len(g),
            "mean_prob": round(float(g[prob_col].mean()), 4),
            "realised": round(float(g[outcome_col].mean()), 4),
            "gap": round(float(g[prob_col].mean() - g[outcome_col].mean()), 4),
            "ece": round(ece(g[prob_col].to_numpy(), g[outcome_col].to_numpy(),
                             bins=bins), 4),
        }
    if group_col is None:
        return pd.DataFrame([{"group": "ALL", **one(df)}])
    rows = [{"group": str(k), **one(g)}
            for k, g in df.groupby(group_col, observed=True) if len(g) >= 50]
    return pd.DataFrame(rows)
