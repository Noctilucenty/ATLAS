"""Shadow evaluation of untraded strategy variants.

These variants are scored from probabilities recorded in real time, so the
gating and clustering must match the live/registered logic exactly - otherwise
a variant would look better or worse than it would actually have performed.
"""

import pytest

from shadow_eval import action_for, count_clusters, expected_value, score_variant

PURGE = 15 * 60


def test_action_matches_the_ev_gate_arithmetic():
    # EV gate: a signal needs p >= (1+margin)/(1+payout).
    # payout 0.87, margin 0.03 -> p >= 0.5508
    assert action_for(0.5600, 0.87, 0.03) == "call"
    assert action_for(0.5400, 0.87, 0.03) is None
    assert action_for(0.4400, 0.87, 0.03) == "put"
    assert action_for(0.4600, 0.87, 0.03) is None


def test_higher_gate_admits_strictly_fewer_signals():
    p, pay = 0.5600, 0.87
    assert action_for(p, pay, 0.02) is not None
    assert action_for(p, pay, 0.10) is None


def test_expected_value_is_symmetric_about_a_half():
    assert expected_value(0.60, 0.87) == pytest.approx(expected_value(0.40, 0.87))


def test_count_clusters_chains_like_forward_eval():
    from forward_eval import cluster_stats

    ts = [100, 400, 1300, 1400, 9000]
    real = cluster_stats([("EURUSD", t, True, 0.87) for t in ts], PURGE)
    ind, clus = count_clusters(ts, PURGE)
    assert clus == real["clusters"]
    assert ind == len(ts)


def _cycle(ts, rows):
    return {"ts": ts, "rows": rows}


def test_score_variant_counts_wins_against_the_label():
    cycles = [_cycle(1000, [
        {"a": "EURUSD", "p": 0.60, "bar": 1000, "pay": 0.87, "open": True},
        {"a": "GBPUSD", "p": 0.40, "bar": 1000, "pay": 0.87, "open": True},
    ])]
    labels = {("EURUSD", 1000): 1.0,   # rose -> call wins
              ("GBPUSD", 1000): 1.0}   # rose -> put loses
    out = score_variant(cycles, labels, 0.03, None, PURGE)
    assert out["gated"] == 2
    assert out["win_rate"] == pytest.approx(0.5)
    assert out["breakeven"] == pytest.approx(1 / 1.87, abs=1e-4)


def test_closed_markets_and_missing_payouts_are_skipped():
    cycles = [_cycle(1000, [
        {"a": "EURUSD", "p": 0.60, "bar": 1000, "pay": 0.87, "open": False},
        {"a": "GBPUSD", "p": 0.60, "bar": 1000, "pay": None, "open": True},
    ])]
    assert score_variant(cycles, {}, 0.03, None, PURGE)["gated"] == 0


def test_unlabelled_rows_are_unresolved_not_silently_dropped():
    """A row with no candle label must be COUNTED as unresolved; dropping it
    quietly would inflate the apparent win rate."""
    cycles = [_cycle(1000, [
        {"a": "EURUSD", "p": 0.60, "bar": 1000, "pay": 0.87, "open": True},
    ])]
    out = score_variant(cycles, {}, 0.03, None, PURGE)
    assert out["gated"] == 0 and out["unresolved"] == 1
    assert out["win_rate"] is None


def test_universe_filter_excludes_outsiders():
    cycles = [_cycle(1000, [
        {"a": "EURUSD", "p": 0.60, "bar": 1000, "pay": 0.87, "open": True},
        {"a": "EURCHF", "p": 0.60, "bar": 1000, "pay": 0.87, "open": True},
    ])]
    labels = {("EURUSD", 1000): 1.0, ("EURCHF", 1000): 1.0}
    everything = score_variant(cycles, labels, 0.03, None, PURGE)
    frozen = score_variant(cycles, labels, 0.03, {"EURUSD"}, PURGE)
    assert everything["gated"] == 2
    assert frozen["gated"] == 1


def test_a_lower_gate_never_admits_fewer_trades():
    """Monotonicity - the property that makes the gate a dial at all."""
    rows = [{"a": "EURUSD", "p": 0.50 + i / 200, "bar": 1000 + 60 * i,
             "pay": 0.87, "open": True} for i in range(20)]
    labels = {("EURUSD", 1000 + 60 * i): 1.0 for i in range(20)}
    loose = score_variant([_cycle(1, rows)], labels, 0.02, None, PURGE)
    tight = score_variant([_cycle(1, rows)], labels, 0.05, None, PURGE)
    assert loose["gated"] >= tight["gated"]
