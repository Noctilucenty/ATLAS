"""Broker expiry snapping - the finding that explained label fidelity.

client.buy(..., 15) does NOT create a contract expiring 15 minutes later. The
vendored expiration.py picks quarter-hour boundaries at least 5 minutes out
and takes whichever is closest to the requested duration. Measured 2026-07-28,
scoring the demo trial on that expiry instead of bar+15 raised broker/candle
agreement from 10/13 to 13/13 - so this arithmetic is now load-bearing for the
project's most important measurement.
"""

import datetime as dt

import pytest

from mission_control import snapped_expiry_ts


def _utc(y, mo, d, h, mi, s=0) -> int:
    return int(dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp())


def _hhmm(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%H:%M")


def test_expiry_always_lands_on_a_quarter_hour():
    for minute in range(0, 60):
        ts = _utc(2026, 7, 28, 5, minute, 19)
        out = dt.datetime.fromtimestamp(snapped_expiry_ts(ts), dt.timezone.utc)
        assert out.minute % 15 == 0, f"{minute} -> {out}"
        assert out.second == 0


def test_the_real_0509_case_that_flipped_a_disagreement():
    """The live trade at 05:09:19 settled at 05:30, not 05:24 - a six-minute
    difference that reversed the outcome."""
    ts = _utc(2026, 7, 28, 5, 9, 19)
    assert _hhmm(snapped_expiry_ts(ts)) == "05:30"


def test_expiry_is_never_less_than_five_minutes_away():
    """The library excludes boundaries under 5 minutes out; picking one would
    mis-time settlement badly on trades placed just before the quarter."""
    for minute in range(0, 60):
        ts = _utc(2026, 7, 28, 9, minute, 30)
        assert snapped_expiry_ts(ts) - ts > 300


def test_picks_the_boundary_closest_to_the_requested_duration():
    # 06:00:00 + 15 min target = 06:15 exactly, and 06:15 is >5 min out
    assert _hhmm(snapped_expiry_ts(_utc(2026, 7, 28, 6, 0, 0))) == "06:15"
    # 06:12 -> 06:15 is only 3 min out (excluded), so 06:30 (18 min) wins
    assert _hhmm(snapped_expiry_ts(_utc(2026, 7, 28, 6, 12, 0))) == "06:30"


def test_differs_from_the_naive_convention_most_of_the_time():
    """If the two rules agreed nearly always, this finding would not matter.
    They differ for the large majority of entry minutes."""
    differing = 0
    for minute in range(60):
        ts = _utc(2026, 7, 28, 10, minute, 19)
        naive = ((ts // 60) * 60) + 15 * 60
        if snapped_expiry_ts(ts) != naive:
            differing += 1
    assert differing > 45, f"only {differing}/60 differ"


@pytest.mark.parametrize("duration,expected_minute_multiple", [(15, 15)])
def test_duration_parameter_is_honoured(duration, expected_minute_multiple):
    ts = _utc(2026, 7, 28, 3, 7, 0)
    out = dt.datetime.fromtimestamp(snapped_expiry_ts(ts, duration),
                                    dt.timezone.utc)
    assert out.minute % expected_minute_multiple == 0
