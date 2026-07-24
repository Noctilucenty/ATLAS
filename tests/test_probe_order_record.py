"""Tests for the raw order-record probe's pure helper."""

from probe_order_record import find_price_fields


def test_finds_nested_price_like_numeric_fields():
    payload = {
        "msg": {"closed_options": [
            {"id": [7], "win": "win", "value": 1.10025,
             "instrument_strike_value": 1100250000,
             "amount": 1.0, "note": "not a price"},
        ]},
        "betinfo": {"deal": {"close_quote": 1.10041, "open_quote": 1.10025}},
    }
    found = find_price_fields(payload)
    assert found["msg.closed_options[0].value"] == 1.10025
    assert found["msg.closed_options[0].instrument_strike_value"] == 1100250000
    assert found["betinfo.deal.close_quote"] == 1.10041
    assert found["betinfo.deal.open_quote"] == 1.10025
    # non-price fields and non-numerics are not reported
    assert not any("note" in k for k in found)
    assert not any("win" in k for k in found)


def test_empty_and_scalar_payloads_are_safe():
    assert find_price_fields({}) == {}
    assert find_price_fields([]) == {}
    assert find_price_fields({"a": None, "b": "text"}) == {}
