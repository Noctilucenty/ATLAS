"""Tail calibration: promised vs realized at the gate.

External research (2026-07-28) established that global Brier skill does NOT
bound win rate at a threshold - realized win rate is the mean selected
confidence plus the tail calibration error. So this arithmetic is what
actually predicts profitability, and it must be right.
"""

import pytest

from tail_calibration import calibration_gap, confidence, day_clustered_ci


def test_confidence_is_distance_from_a_coin_flip():
    assert confidence(0.56) == pytest.approx(0.56)
    assert confidence(0.44) == pytest.approx(0.56)   # a put is equally confident
    assert confidence(0.50) == pytest.approx(0.50)


def test_a_perfectly_calibrated_tail_has_zero_gap():
    # 100 trades promised 0.60, exactly 60 win
    rows = [{"q": 0.60, "won": i < 60} for i in range(100)]
    out = calibration_gap(rows)
    assert out["promised"] == pytest.approx(0.60)
    assert out["realized"] == pytest.approx(0.60)
    assert out["gap"] == pytest.approx(0.0)


def test_overconfidence_shows_as_a_negative_gap():
    """The failure mode that turns a positive-EV gate into a losing one."""
    rows = [{"q": 0.60, "won": i < 52} for i in range(100)]
    out = calibration_gap(rows)
    assert out["gap"] == pytest.approx(-0.08)
    assert out["gap"] < 0


def test_underconfidence_shows_as_a_positive_gap():
    rows = [{"q": 0.55, "won": i < 65} for i in range(100)]
    assert calibration_gap(rows)["gap"] == pytest.approx(0.10)


def test_empty_input_is_safe():
    out = calibration_gap([])
    assert out["n"] == 0 and out["gap"] is None


def test_promised_averages_mixed_confidences():
    rows = [{"q": 0.55, "won": True}, {"q": 0.65, "won": False}]
    out = calibration_gap(rows)
    assert out["promised"] == pytest.approx(0.60)
    assert out["realized"] == pytest.approx(0.50)


def test_day_clustering_widens_the_interval_versus_one_day():
    """Trades within a day share overlapping windows, so resampling days must
    NOT produce the tight interval that resampling trades would."""
    # 5 days, one all-win day and one all-loss day -> genuinely uncertain
    rows = []
    for d, wins in enumerate([True, False, True, False, True]):
        for _ in range(20):
            rows.append({"q": 0.55, "won": wins, "day": f"2026-07-0{d + 1}"})
    ci = day_clustered_ci(rows, iters=500)
    assert ci is not None
    lo, hi = ci
    assert hi - lo > 0.3, f"interval implausibly tight: {ci}"


def test_single_day_returns_no_interval():
    """One cluster cannot support an interval - saying so beats inventing one."""
    rows = [{"q": 0.55, "won": True, "day": "2026-07-28"} for _ in range(50)]
    assert day_clustered_ci(rows) is None


def test_interval_is_deterministic_for_a_given_seed():
    rows = [{"q": 0.55, "won": i % 2 == 0, "day": f"2026-07-{10 + i % 4}"}
            for i in range(40)]
    assert day_clustered_ci(rows, iters=300) == day_clustered_ci(rows, iters=300)
