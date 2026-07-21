import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_report import (  # noqa: E402
    build_report,
    parse_cycle_statuses,
    payout_coverage_estimate,
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


# ---------------- coverage estimation (mirrors causal join semantics) ----------------

def test_coverage_counts_only_causal_fresh_snapshots():
    snapshots = [1000, 5000]
    candles = [900, 1500, 5000, 5000 + 7200, 5000 + 7201]
    # 900: no prior snapshot. 1500: covered by 1000. 5000: at == covered.
    # 5000+7200: exactly max age, covered. 5000+7201: stale.
    assert payout_coverage_estimate(candles, snapshots) == 3 / 5

def test_coverage_never_uses_future_snapshots():
    assert payout_coverage_estimate([999], [1000]) == 0.0

def test_coverage_empty_inputs():
    assert payout_coverage_estimate([], [1000]) is None
    assert payout_coverage_estimate([1000], []) == 0.0


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


def test_build_report_flags_missing_required_payout_key(tmp_path):
    now = 1_800_000_000
    conn = open_db(tmp_path / "m.duckdb")
    store_dataset(conn, "EURUSD", 60, make_candles(120, now - 7200), [])
    store_dataset(conn, "EURUSD-OTC", 60, make_candles(120, now - 7200), [])
    snapshot(conn, now - 3600, "EURUSD-op")   # OTC key deliberately absent
    log = "=== cycle exit status: 0 ===\n"

    report = build_report(conn, log, now)
    spot = report["instruments"]["EURUSD"]
    otc = report["instruments"]["EURUSD-OTC"]
    assert spot["payout_key_in_latest_batch"] is True
    assert otc["payout_key_in_latest_batch"] is False
    assert report["healthy"] is False          # missing required key -> unhealthy
    assert spot["candles"] == 120 and spot["gaps"] == 0 and spot["conflicts"] == 0
    assert spot["distinct_utc_days"] >= 1
    assert 0 < spot["prospective_coverage_48h"] <= 1


def test_build_report_healthy_when_all_keys_present_and_cycles_clean(tmp_path):
    now = 1_800_000_000
    conn = open_db(tmp_path / "m.duckdb")
    store_dataset(conn, "EURUSD", 60, make_candles(60, now - 3600), [])
    store_dataset(conn, "EURUSD-OTC", 60, make_candles(60, now - 3600), [])
    snapshot(conn, now - 600, "EURUSD-op")
    snapshot(conn, now - 600, "EURUSD-OTC")
    report = build_report(conn, "=== cycle exit status: 0 ===\n", now)
    assert report["healthy"] is True

def test_build_report_unhealthy_on_failed_last_cycle(tmp_path):
    now = 1_800_000_000
    conn = open_db(tmp_path / "m.duckdb")
    store_dataset(conn, "EURUSD", 60, make_candles(60, now - 3600), [])
    store_dataset(conn, "EURUSD-OTC", 60, make_candles(60, now - 3600), [])
    snapshot(conn, now - 600, "EURUSD-op")
    snapshot(conn, now - 600, "EURUSD-OTC")
    report = build_report(conn, "=== cycle exit status: 2 ===\n", now)
    assert report["healthy"] is False
