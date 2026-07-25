"""Signals outside the frozen training universe must be counted separately.

The deployed bundle was trained on 16 instruments; the live list is 39. A
win rate that blends validated and unvalidated instruments is not the
deployed strategy's win rate, and the registered candles verdict excludes
the outsiders anyway.
"""

import mission_control as mc

UNIVERSE = {"EURUSD", "GBPUSD", "USDJPY"}


def test_splits_in_and_out_of_universe():
    signals = [
        {"asset": "EURUSD"}, {"asset": "GBPUSD"}, {"asset": "GBPUSD"},
        {"asset": "EURCHF"}, {"asset": "EURCHF"}, {"asset": "XAUUSD"},
    ]
    out = mc.universe_split(signals, UNIVERSE)
    assert out["universe_size"] == 3
    assert out["in_universe"] == 3
    assert out["out_of_universe"] == 3
    assert out["out_share"] == 0.5
    assert out["out_by_asset"] == {"EURCHF": 2, "XAUUSD": 1}


def test_all_in_universe_reports_zero_outside():
    out = mc.universe_split([{"asset": "EURUSD"}], UNIVERSE)
    assert out["out_of_universe"] == 0
    assert out["out_by_asset"] == {}


def test_missing_universe_degrades_gracefully():
    out = mc.universe_split([{"asset": "EURUSD"}], set())
    assert out["universe_size"] == 0
    assert "unavailable" in out["note"]


def test_empty_signal_list_has_no_share():
    out = mc.universe_split([], UNIVERSE)
    assert out["in_universe"] == 0 and out["out_share"] is None
