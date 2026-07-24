"""Pure-function tests for the audit-repair tools (gap catchup + orphan
settlement)."""

from catchup_gaps import plan_catchup
from settle_missing import find_unsettled

H = 3600


def test_plan_catchup_skips_fresh_assets():
    now = 1_000_000
    needy, hours = plan_catchup({"EURUSD": now - H, "GBPUSD": now - 2 * H}, now)
    assert needy == [] and hours == 0.0


def test_plan_catchup_heals_worst_gap_for_all_needy():
    now = 1_000_000
    latest = {"EURUSD": now - 10 * H,   # 10h hole
              "GBPUSD": now - 4 * H,    # 4h hole
              "USDJPY": now - H}        # fresh
    needy, hours = plan_catchup(latest, now)
    assert set(needy) == {"EURUSD", "GBPUSD"}
    assert hours == 11.0  # worst gap + 1h margin


def test_plan_catchup_caps_and_defaults():
    now = 1_000_000
    latest = {"OLD": now - 10_000 * H, "NEW_ASSET": None}
    needy, hours = plan_catchup(latest, now)
    assert set(needy) == {"OLD", "NEW_ASSET"}
    assert hours == 168.0  # capped at a week


def test_find_unsettled_matches_by_order_id():
    rows = [
        {"ts": 1, "order_id": 11},                      # settled below
        {"ts": 1, "order_id": 11, "settled": True},
        {"ts": 2, "order_id": 22},                      # orphan
        {"ts": 3},                                      # paper row, no order
        {"ts": 4, "order_id": None},                    # refused order
    ]
    orphans = find_unsettled(rows)
    assert [r["order_id"] for r in orphans] == [22]
