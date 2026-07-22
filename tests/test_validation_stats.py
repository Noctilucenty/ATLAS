import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation_stats import (  # noqa: E402
    deflated_win_rate,
    ece,
    expected_max_z,
    min_track_record,
    pbo_cscv,
    reliability_table,
)


# ---------------- PBO ----------------

def test_pbo_high_for_pure_noise():
    """Random configs: the IS-best is meaningless, so it should fall below
    the OOS median about half the time."""
    rng = np.random.default_rng(0)
    perf = pd.DataFrame(rng.normal(0.5, 0.02, size=(20, 8)))
    result = pbo_cscv(perf)
    assert 0.3 < result["pbo"] < 0.7
    assert result["n_combinations"] == 70  # C(8,4)


def test_pbo_low_for_dominant_config():
    """One config genuinely better in every fold: choosing it IS should
    keep working OOS, so PBO should be near zero."""
    rng = np.random.default_rng(1)
    perf = pd.DataFrame(rng.normal(0.50, 0.01, size=(20, 8)))
    perf.iloc[3] = rng.normal(0.60, 0.01, size=8)  # dominant everywhere
    assert pbo_cscv(perf)["pbo"] < 0.1


def test_pbo_rejects_odd_or_tiny_fold_counts():
    with pytest.raises(ValueError):
        pbo_cscv(pd.DataFrame(np.zeros((5, 7))))
    with pytest.raises(ValueError):
        pbo_cscv(pd.DataFrame(np.zeros((5, 2))))


# ---------------- deflated win rate ----------------

def test_expected_max_z_grows_with_trials():
    zs = [expected_max_z(n) for n in (1, 10, 100, 1000)]
    assert zs[0] == 0.0
    assert zs == sorted(zs)
    assert 2.0 < zs[2] < 3.0  # max of 100 normals ~ 2.5


def test_deflation_penalises_many_trials():
    few = deflated_win_rate(wins=580, n=1000, breakeven=0.5348, n_trials=1)
    many = deflated_win_rate(wins=580, n=1000, breakeven=0.5348, n_trials=1000)
    assert few["p_deflated"] < many["p_deflated"]
    assert few["z_raw"] == many["z_raw"]  # raw evidence unchanged


def test_strong_edge_survives_honest_deflation():
    # The decade holdout scale: 71.5% on 1309 trades vs 98 trials attempted.
    r = deflated_win_rate(wins=936, n=1309, breakeven=0.5348, n_trials=98)
    assert r["passes_05"] is True


def test_marginal_edge_dies_under_deflation():
    # 55% on 400 trades looks nominally significant but was picked from 500 tries.
    r = deflated_win_rate(wins=220, n=400, breakeven=0.5348, n_trials=500)
    assert r["passes_05"] is False


# ---------------- MinTRL ----------------

def test_min_track_record_scales_inversely_with_edge():
    close = min_track_record(0.55, 0.5348)
    far = min_track_record(0.70, 0.5348)
    assert far < close
    assert min_track_record(0.53, 0.5348) == -1  # below break-even: never


def test_min_track_record_matches_binomial_logic():
    n = min_track_record(0.60, 0.5348, alpha=0.05)
    # At exactly n trades, the one-sided z against break-even clears 1.645.
    z = (0.60 - 0.5348) / np.sqrt(0.60 * 0.40 / n)
    assert z >= 1.645


# ---------------- calibration ----------------

def test_ece_near_zero_for_perfect_calibration():
    rng = np.random.default_rng(2)
    prob = rng.uniform(0.3, 0.7, 20000)
    outcome = (rng.uniform(size=20000) < prob).astype(float)
    assert ece(prob, outcome) < 0.02


def test_ece_large_for_overconfident_model():
    prob = np.full(5000, 0.9)
    outcome = np.zeros(5000)
    outcome[:2500] = 1.0  # true rate 0.5 vs claimed 0.9
    assert ece(prob, outcome) > 0.3


def test_reliability_table_groups_and_gaps():
    df = pd.DataFrame({
        "p": [0.6] * 100 + [0.7] * 100,
        "won": [1.0] * 60 + [0.0] * 40 + [1.0] * 70 + [0.0] * 30,
        "pair": ["EURUSD"] * 100 + ["GBPUSD"] * 100,
    })
    table = reliability_table(df, "p", "won", "pair")
    assert set(table["group"]) == {"EURUSD", "GBPUSD"}
    assert (table["gap"].abs() < 0.01).all()  # both perfectly calibrated
