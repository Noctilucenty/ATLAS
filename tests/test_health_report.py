import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_report import (  # noqa: E402
    MIN_DAY_CANDLES,
    build_report,
    day_breakdown,
    parse_cycle_statuses,
    payout_coverage_counts,
)
from storage import open_db, store_dataset  # noqa: E402


# ---------------- cycle-status parsing ----------------

def test_parse_cycle_statuses_reads_recent_history():
    log = (
        "=== cycle a ===\nnoise\n=== cycle exit status: 0 ===\n"
        "=== cycle b ===\n=== cycle exit status: 1 ===\n"
        "=== cycle c ===\n=== cycle exit status: 0 ===\n"
    )
    assert parse_cycle_statuses(log) == [0, 1, 0]
    assert parse_cycle_statuses(log, last=2) == [1, 0]
    assert parse_cycle_statuses("") == []

def test_current_cycle_status_is_included_not_one_behind():
    log = "=== cycle exit status: 0 ===\n"
    # The running cycle's line is appended AFTER the report; passing it
    # explicitly means health.json reflects the cycle that just collected.
    assert parse_cycle_statuses(log, current_status=2) == [0, 2]
    assert parse_cycle_statuses("", current_status=0) == [0]


# ---------------- coverage counts (honest denominators) ----------------

def test_coverage_counts_only_causal_fresh_snapshots():
    snapshots = [1000, 5000]
    candles = [900, 1500, 5000, 5000 + 7200, 5000 + 7201]
    # 900: no prior snapshot. 1500: covered by 1000. 5000: at == covered.
    # 5000+7200: exactly max age, covered. 5000+7201: stale.
    assert payout_coverage_counts(candles, snapshots) == (3, 5)

def test_coverage_never_uses_future_snapshots():
    assert payout_coverage_counts([999], [1000]) == (0, 1)

def test_coverage_denominator_is_collected_candles_not_wall_clock():
    # 68 covered of 160 collected must report exactly those counts - the
    # ratio's denominator is what we collected, never 2880 window minutes.
    snapshots = [0]
    candles = list(range(60, 68 * 60 + 1, 60)) + list(range(1_000_000, 1_000_000 + 92 * 60, 60))
    covered, collected = payout_coverage_counts(candles, snapshots, max_age_s=7200)
    assert collected == 160
    assert covered == 68  # only the first 68 candles are within max age of the snapshot


# ---------------- eligible days vs touched dates ----------------

def day_ts(day_offset: int, minutes: int, start_minute: int = 0) -> list[int]:
    base = 1_784_505_600 + day_offset * 86400  # 2026-07-19T00:00:00Z, a UTC midnight
    return [base + (start_minute + m) * 60 for m in range(minutes)]

def test_touched_dates_are_not_eligible_days():
    # 30 minutes before midnight + 30 after: touches 2 dates, 0 eligible days.
    candles = day_ts(0, 30, start_minute=1410) + day_ts(1, 30)
    snapshots = [candles[0] - 60]
    days = day_breakdown(candles, snapshots)
    assert len(days) == 2
    assert all(not v["eligible"] for v in days.values())
    assert [v["candles"] for v in days.values()] == [30, 30]

def test_full_covered_day_is_eligible():
    candles = day_ts(0, 1440)
    snapshots = [candles[0] - 60 + i * 3600 for i in range(30)]  # hourly snapshots
    days = day_breakdown(candles, snapshots)
    (day,) = days.values()
    assert day["candles"] == 1440
    assert day["payout_coverage"] >= 0.95
    assert day["eligible"] is True

def test_full_day_without_payout_coverage_is_not_eligible():
    candles = day_ts(0, 1440)
    days = day_breakdown(candles, snapshot_ts=[])
    (day,) = days.values()
    assert day["candles"] >= MIN_DAY_CANDLES
    assert day["payout_coverage"] == 0.0
    assert day["eligible"] is False


# ---------------- full report ----------------

def make_candles(n, start, interval=60):
    return [
        {
            "from_ts": start + i * interval,
            "to_ts": start + (i + 1) * interval,
            "open": 1.1, "high": 1.12, "low": 1.09, "close": 1.11, "volume": 1.0,
        }
        for i in range(n)
    ]


def snapshot(conn, ts, quote_key, kind="turbo", payout=0.85):
    conn.execute(
        """INSERT INTO payout_snapshots (ts_utc, ts_epoch, asset, kind, payout, source)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.fromtimestamp(ts, timezone.utc), ts, quote_key, kind, payout, "test"),
    )


def seeded_db(tmp_path, now):
    conn = open_db(tmp_path / "m.duckdb")
    store_dataset(conn, "EURUSD", 60, make_candles(120, now - 7200), [])
    store_dataset(conn, "EURUSD-OTC", 60, make_candles(120, now - 7200), [])
    return conn


def test_report_exposes_counts_and_estimate_semantics(tmp_path):
    now = 1_800_000_000
    conn = seeded_db(tmp_path, now)
    snapshot(conn, now - 3600, "EURUSD-op")
    snapshot(conn, now - 3600, "EURUSD-OTC")
    report = build_report(conn, "=== cycle exit status: 0 ===\n", now, current_cycle_status=0)
    spot = report["instruments"]["EURUSD"]
    assert spot["collected_candles_48h"] == 120
    assert spot["payout_covered_candles_48h"] > 0
    ratio = spot["payout_coverage_on_collected_candles_48h"]
    assert ratio == round(spot["payout_covered_candles_48h"] / 120, 4)
    assert "ESTIMATE" in report["semantics"]
    assert "frozen walk-forward" in report["semantics"]
    assert report["eligible_day_rule"]["min_candles"] == MIN_DAY_CANDLES
    assert spot["utc_dates_touched"] >= 1
    assert spot["eligible_days"] == 0  # 120 candles cannot make an eligible day
    assert report["cycles"]["recent_exit_statuses"] == [0, 0]

def test_report_flags_missing_required_payout_key(tmp_path):
    now = 1_800_000_000
    conn = seeded_db(tmp_path, now)
    snapshot(conn, now - 3600, "EURUSD-op")   # OTC key deliberately absent
    report = build_report(conn, "", now, current_cycle_status=0)
    assert report["instruments"]["EURUSD"]["payout_key_in_latest_batch"] is True
    assert report["instruments"]["EURUSD-OTC"]["payout_key_in_latest_batch"] is False
    assert report["healthy"] is False

def test_warnings_do_not_hide_behind_healthy(tmp_path):
    from instruments import INSTRUMENTS

    now = 1_800_000_000
    conn = open_db(tmp_path / "m.duckdb")
    # Gap between two stored ranges + stale latest candle (4h old); every
    # registered instrument gets data so only warnings remain.
    store_dataset(conn, "EURUSD", 60, make_candles(60, now - 90000), [])
    for asset, spec in INSTRUMENTS.items():
        store_dataset(conn, asset, 60, make_candles(60, now - 18000), [])
        snapshot(conn, now - 15000, spec.quote_key, kind=spec.option_kind)
    report = build_report(conn, "=== cycle exit status: 0 ===\n", now, current_cycle_status=0)
    assert report["healthy"] is True  # collection itself is fine...
    joined = " | ".join(report["warnings"])
    assert "gap" in joined
    assert "stale" in joined

def test_no_parsed_statuses_is_warned(tmp_path):
    now = 1_800_000_000
    conn = seeded_db(tmp_path, now)
    snapshot(conn, now - 3600, "EURUSD-op")
    snapshot(conn, now - 3600, "EURUSD-OTC")
    report = build_report(conn, "", now)
    assert any("no cycle statuses" in w for w in report["warnings"])

def test_failed_current_cycle_makes_unhealthy(tmp_path):
    now = 1_800_000_000
    conn = seeded_db(tmp_path, now)
    snapshot(conn, now - 3600, "EURUSD-op")
    snapshot(conn, now - 3600, "EURUSD-OTC")
    report = build_report(conn, "=== cycle exit status: 0 ===\n", now, current_cycle_status=2)
    assert report["healthy"] is False
