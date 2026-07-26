"""Probe orchestration timeline.

The one mistake worth engineering out is starting the quote recorder too
late - it silently costs the settlement-rule attribution while still producing
a plausible-looking result. So the recorder must always outlive the final
expiry, and settlement must always come after it.
"""

import pytest

from probe_run import EXPIRY_MINUTES, plan


def test_recorder_outlives_the_last_expiry():
    t = plan(trades=6, spacing=120)
    assert t["recorder_s"] > t["last_expiry_s"], (
        "recorder would stop before the last trade expires - the trimmed "
        "settlement label could not be computed")


def test_settlement_happens_after_the_last_expiry():
    t = plan(trades=6, spacing=120)
    assert t["settle_wait_s"] > t["last_expiry_s"]


def test_last_expiry_accounts_for_both_placement_and_the_expiry_itself():
    t = plan(trades=6, spacing=120)
    assert t["placement_s"] == 5 * 120
    assert t["last_expiry_s"] == 5 * 120 + EXPIRY_MINUTES * 60


def test_single_trade_has_no_placement_span():
    t = plan(trades=1, spacing=120)
    assert t["placement_s"] == 0
    assert t["last_expiry_s"] == EXPIRY_MINUTES * 60
    assert t["recorder_s"] > t["last_expiry_s"]


def test_zero_trades_does_not_produce_negative_time():
    t = plan(trades=0, spacing=120)
    assert t["placement_s"] == 0


@pytest.mark.parametrize("trades,spacing", [(2, 30), (6, 120), (12, 300)])
def test_ordering_invariants_hold_for_any_shape(trades, spacing):
    t = plan(trades, spacing)
    assert t["placement_s"] <= t["last_expiry_s"] < t["settle_wait_s"]
    assert t["settle_wait_s"] < t["recorder_s"] + 1
