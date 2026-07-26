"""Settlement-rule attribution math.

If IQ settles on a trimmed average of pre-expiry quotes rather than a single
close (Nadex publishes exactly such a rule), then a broker disagreement is an
averaging artefact, not a markup. Those have opposite consequences, so the
trimming must be right - and must REFUSE to guess when it has too few samples,
because inventing a settlement price would silently fabricate the finding.
"""

import json

import pytest

from quote_recorder import (NADEX_DISCARD_EACH_SIDE, NADEX_SAMPLE_COUNT,
                            read_quotes, trimmed_settlement)


def test_drops_three_highest_and_three_lowest_of_ten():
    # 1..10: dropping 1,2,3 and 8,9,10 leaves 4,5,6,7 -> mean 5.5
    quotes = [float(i) for i in range(1, 11)]
    assert trimmed_settlement(quotes) == pytest.approx(5.5)


def test_uses_only_the_LAST_ten_quotes():
    """Earlier quotes must not influence settlement - only the pre-expiry
    window counts."""
    quotes = [999.0] * 20 + [float(i) for i in range(1, 11)]
    assert trimmed_settlement(quotes) == pytest.approx(5.5)


def test_outliers_are_exactly_what_gets_discarded():
    """The point of trimming: spikes at either extreme cannot move settlement."""
    clean = [1.1000] * 10
    spiked = [1.1000] * 4 + [1.5, 1.6, 1.7] + [0.5, 0.6, 0.7]
    assert trimmed_settlement(clean) == pytest.approx(1.1000)
    # 3 high and 3 low spikes are removed, leaving the four 1.1000 samples
    assert trimmed_settlement(spiked) == pytest.approx(1.1000)


def test_refuses_to_settle_on_too_few_samples():
    """Averaging 2 quotes would INVENT a settlement price and fabricate the
    attribution. None is the honest answer."""
    assert trimmed_settlement([]) is None
    assert trimmed_settlement([1.1] * 2) is None
    assert trimmed_settlement([1.1] * 9) is None
    assert trimmed_settlement([1.1] * 10) is not None


def test_discard_wider_than_the_window_is_refused():
    assert trimmed_settlement([1.1] * 10, sample_count=10, discard=5) is None


def test_nadex_constants_match_the_published_rule():
    assert NADEX_SAMPLE_COUNT == 10
    assert NADEX_DISCARD_EACH_SIDE == 3


def test_read_quotes_filters_by_window_and_survives_torn_lines(tmp_path,
                                                              monkeypatch):
    import quote_recorder

    monkeypatch.setattr(quote_recorder, "LOGS", tmp_path)
    path = tmp_path / "quotes_EURUSD.jsonl"
    path.write_text(
        json.dumps({"ts": 100, "q": 1.10}) + "\n"
        + json.dumps({"ts": 200, "q": 1.11}) + "\n"
        + '{"ts": 250, "q":\n'                       # torn mid-append
        + json.dumps({"ts": 300, "q": 1.12}) + "\n"
        + json.dumps({"ts": 400, "q": "bad"}) + "\n"  # non-numeric quote
        , encoding="utf-8")
    got = read_quotes("EURUSD", 150, 350)
    assert got == [(200, 1.11), (300, 1.12)]


def test_read_quotes_returns_sorted_by_time(tmp_path, monkeypatch):
    import quote_recorder

    monkeypatch.setattr(quote_recorder, "LOGS", tmp_path)
    (tmp_path / "quotes_EURUSD.jsonl").write_text(
        json.dumps({"ts": 300, "q": 3.0}) + "\n"
        + json.dumps({"ts": 100, "q": 1.0}) + "\n"
        + json.dumps({"ts": 200, "q": 2.0}) + "\n", encoding="utf-8")
    assert [t for t, _ in read_quotes("EURUSD", 0, 999)] == [100, 200, 300]


def test_missing_log_is_empty_not_an_error(tmp_path, monkeypatch):
    import quote_recorder

    monkeypatch.setattr(quote_recorder, "LOGS", tmp_path)
    assert read_quotes("NOSUCH", 0, 999) == []
