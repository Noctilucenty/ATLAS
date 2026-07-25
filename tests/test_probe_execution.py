"""Tests for the direct execution probe's pure helpers.

The probe places real (demo) orders, so its label and summary logic must be
right before it is ever run with --confirm.
"""

import pytest

from probe_execution import label_from_prices, summarize


def test_label_follows_direction_and_registered_convention():
    # strike = next bar's open, exercise = close 15 bars later
    assert label_from_prices(1.1000, 1.1010, "call") == "win"
    assert label_from_prices(1.1000, 1.0990, "call") == "loose"
    assert label_from_prices(1.1000, 1.0990, "put") == "win"
    assert label_from_prices(1.1000, 1.1010, "put") == "loose"
    assert label_from_prices(1.1000, 1.1000, "call") == "equal"
    assert label_from_prices(1.1000, 1.1000, "put") == "equal"


def test_summarize_counts_agreement():
    rows = [
        {"order_id": 1, "broker_result": "win", "candle_label": "win"},
        {"order_id": 2, "broker_result": "loose", "candle_label": "loose"},
        {"order_id": 3, "broker_result": "win", "candle_label": "loose"},
        {"order_id": 4},                       # placed, not settled
        {"order_error": "rejected"},           # never placed
    ]
    out = summarize(rows)
    assert out["placed"] == 4
    assert out["settled"] == 3
    assert out["agree"] == 2 and out["disagree"] == 1
    assert out["agreement_rate"] == pytest.approx(0.6667, abs=1e-4)


def test_summarize_flags_markup_when_strikes_deviate():
    rows = [
        {"order_id": 1, "broker_result": "win", "candle_label": "win",
         "strike_delta_pips": 0.02},
        {"order_id": 2, "broker_result": "win", "candle_label": "win",
         "strike_delta_pips": -0.03},
    ]
    out = summarize(rows)
    assert out["strike_samples"] == 2
    assert "MID striking" in out["verdict_hint"]

    rows.append({"order_id": 3, "broker_result": "loose",
                 "candle_label": "win", "strike_delta_pips": 0.45})
    out2 = summarize(rows)
    assert "MARKUP DETECTED" in out2["verdict_hint"]
    assert out2["strike_delta_pips_max_abs"] == pytest.approx(0.45)


def test_summarize_empty_is_safe():
    out = summarize([])
    assert out["placed"] == 0 and out["agreement_rate"] is None
    assert "verdict_hint" not in out


def test_probe_writes_to_its_own_log_not_the_forward_test_log():
    """Isolation is the whole safety argument: forward_eval reads
    logs/live_h2.jsonl, so the probe must never write there."""
    import probe_execution

    assert probe_execution.OUT_PATH.name == "execution_probe.jsonl"
    source = (probe_execution.PROJECT_DIR / "probe_execution.py").read_text(
        encoding="utf-8")
    # the only jsonl path referenced for writing is the probe's own
    assert "live_h2.jsonl" not in source.replace(
        "never to logs/live_h2.jsonl", "")


def test_confirm_is_required_to_place(monkeypatch):
    """Without --confirm the probe must exit non-zero before connecting."""
    import probe_execution

    monkeypatch.setattr("sys.argv", ["probe_execution.py", "--trades", "1"])
    assert probe_execution.main() == 2
