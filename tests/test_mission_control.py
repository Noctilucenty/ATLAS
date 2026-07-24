"""Mission Control core tests - pure functions only, no broker, no
Task Scheduler, no live databases."""

import json

import pytest

import mission_control as mc


# ------------------------------------------------------------- jsonl / split

def test_read_jsonl_skips_torn_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"broken\n{"b": 2}\n')
    rows = mc.read_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_split_signals_separates_settled_duplicates():
    rows = [
        {"ts": 1, "asset": "EURUSD", "order_id": 11},
        {"ts": 1, "asset": "EURUSD", "order_id": 11, "settled": True,
         "result": "win", "profit": 0.87},
        {"ts": 2, "asset": "GBPUSD", "trade_skipped": "otc_below_breakeven"},
        {"ts": 3, "asset": "USDJPY"},
    ]
    parts = mc.split_signals(rows)
    assert len(parts["signals"]) == 3          # settled dup not double counted
    assert len(parts["settled"]) == 1
    assert len(parts["placed"]) == 1
    assert len(parts["skipped_otc"]) == 1


# ------------------------------------------------------------------ EV math

def test_expected_value_matches_binary_economics():
    # p=0.6, payout=0.87: call EV = .6*.87 - .4 = .122
    assert mc.expected_value(0.6, 0.87) == pytest.approx(0.122)
    # symmetric put side: p=0.4 must give the same EV via the put
    assert mc.expected_value(0.4, 0.87) == pytest.approx(0.122)


def test_forward_progress_counts_only_registered_gates():
    signals = [
        # ev = .122 > .04 > .03 ; meta above threshold ; h4 present
        {"p_up": 0.6, "payout": 0.87, "meta_p": 0.65, "h4_p": 0.5},
        # ev = .028 -> below both EV gates, meta irrelevant
        {"p_up": 0.55, "payout": 0.87, "meta_p": 0.99},
    ]
    counts = mc.forward_progress(signals)
    assert counts["H2p ev0.03"] == 1
    assert counts["H2s ev0.04"] == 1
    assert counts["H3 meta0.60"] == 1
    assert counts["H4"] == 1


# ------------------------------------------------------- supervisor parsing

def test_parse_and_churn_detection():
    lines = [
        "[2026-07-24T04:00:00Z] runner launched pid=1",
        "[2026-07-24T04:00:30Z] runner exited (0); relaunching",
        "[2026-07-24T04:01:00Z] runner exited (0); relaunching",
        "[2026-07-24T04:01:30Z] runner exited (0); relaunching",
        "not a log line",
    ]
    events = mc.parse_supervisor_events(lines)
    assert len(events) == 4
    now = events[-1][0] + 60
    assert mc.runner_churn(events, now) == 3
    # outside the window they age out
    assert mc.runner_churn(events, now + mc.CHURN_WINDOW_S + 120) == 0


# ------------------------------------------------------------- health tiers

def _tier(**kw):
    defaults = dict(now=1000_000, heartbeat_ts=1000_000 - 60,
                    task_status="Running", churn_events=0,
                    latest_candle_ts=1000_000 - 120, supervisor_seen=True)
    defaults.update(kw)
    return mc.classify_health(**defaults)


def test_healthy_baseline():
    tier, reasons = _tier()
    assert tier == "HEALTHY" and reasons == []


def test_stale_heartbeat_is_critical():
    tier, reasons = _tier(heartbeat_ts=1000_000 - mc.HEARTBEAT_STALE_S - 1)
    assert tier == "CRITICAL"
    assert any("heartbeat" in r for r in reasons)


def test_task_not_running_is_critical():
    tier, _ = _tier(task_status="Ready")
    assert tier == "CRITICAL"


def test_churn_and_stale_candles_warn_without_flipping_critical():
    tier, reasons = _tier(churn_events=5,
                          latest_candle_ts=1000_000 - mc.CANDLE_STALE_S - 1)
    assert tier == "WARNING"
    assert len(reasons) == 2


def test_tier_exit_codes():
    assert mc.tier_exit_code("HEALTHY") == 0
    assert mc.tier_exit_code("WARNING") == 1
    assert mc.tier_exit_code("CRITICAL") == 2


# ----------------------------------------------------------- label fidelity

def test_label_fidelity_agreement_and_disagreement():
    # Two settled orders on EURUSD spot. Candle closes: entry 1.1000.
    # Trade A: call, settle 1.1010 (up) -> candle win; broker says win  -> agree
    # Trade B: call, settle 1.0990 (dn) -> candle loose; broker says win -> disagree
    settled = [
        {"ts": 600_000, "bar_to_ts": 600_000, "asset": "EURUSD",
         "action": "binary_call", "order_id": 1, "result": "win", "profit": 0.87},
        {"ts": 700_000, "bar_to_ts": 700_000, "asset": "EURUSD",
         "action": "binary_call", "order_id": 2, "result": "win", "profit": 0.87},
    ]

    def fake_closes(asset, ts_list):
        assert asset == "EURUSD"  # candle_asset mapping applied
        table = {600_000: 1.1000, mc._bar_ts(600_000 + mc.EXPIRY_S): 1.1010,
                 700_000: 1.1000, mc._bar_ts(700_000 + mc.EXPIRY_S): 1.0990}
        return {t: table[t] for t in ts_list if t in table}

    out = mc.label_fidelity(settled, closes_fn=fake_closes)
    assert out["settled_orders"] == 2
    assert out["judged"] == 2
    assert out["agree"] == 1 and out["disagree"] == 1
    assert out["agreement_rate"] == 0.5


def test_label_fidelity_missing_candles_are_undetermined():
    settled = [{"ts": 600_000, "bar_to_ts": 600_000, "asset": "EURUSD",
                "action": "binary_put", "order_id": 3, "result": "loose",
                "profit": -1.0}]
    out = mc.label_fidelity(settled, closes_fn=lambda a, ts: {})
    assert out["undetermined"] == 1
    assert out["judged"] == 0
    assert out["agreement_rate"] is None
