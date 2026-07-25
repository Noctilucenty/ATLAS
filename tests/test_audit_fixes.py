"""Regression tests for the 2026-07-24 adversarial-review findings.

Each test pins a defect the review confirmed, so the unattended guard layer
cannot silently regress to failing in the dark.
"""

import json

import pytest

import backup_state
import mission_control as mc
from settle_missing import index_closed_options


# --------------------------------------------- sidecar visibility (systemic)

def test_last_log_entry_reads_newest_stamped_line(tmp_path):
    from datetime import datetime, timezone

    p = tmp_path / "x.log"
    p.write_text("[2026-07-24T01:00:00Z] first\n"
                 "not a stamped line\n"
                 "[2026-07-24T03:30:00Z] second\n")
    ts, line = mc.last_log_entry(p)
    assert line.endswith("second")
    # stamps are UTC, never local time
    assert ts == int(datetime(2026, 7, 24, 3, 30,
                              tzinfo=timezone.utc).timestamp())


def test_last_log_entry_missing_file_is_none(tmp_path):
    assert mc.last_log_entry(tmp_path / "nope.log") == (None, "")


def test_partial_sidecar_failure_is_not_flagged(tmp_path):
    """Quoted-hours assets are shut every weekend, so failed>=1 with other
    assets stored is normal. Flagging it would cry wolf permanently."""
    import datetime as _dt

    p = tmp_path / "extra.log"
    p.write_text("[2026-07-25T03:05:00Z] stored=5 (SpaceX-OTC:239) "
                 "failed=1 (SpaceX-op[SchemaError])\n")
    now = int(_dt.datetime(2026, 7, 25, 3, 30,
                           tzinfo=_dt.timezone.utc).timestamp())
    out = mc.sidecar_health(now, {"extra": (p, 3600)})
    assert out["extra"]["failed"] is False
    assert out["extra"]["stale"] is False


def test_total_sidecar_failure_is_flagged(tmp_path):
    import datetime as _dt

    p = tmp_path / "extra.log"
    p.write_text("[2026-07-25T03:05:00Z] stored=0 () failed=6 (all of them)\n")
    now = int(_dt.datetime(2026, 7, 25, 3, 30,
                           tzinfo=_dt.timezone.utc).timestamp())
    out = mc.sidecar_health(now, {"extra": (p, 3600)})
    assert out["extra"]["failed"] is True


def test_sidecar_health_flags_stale_and_failed(tmp_path):
    fresh = tmp_path / "fresh.log"
    old = tmp_path / "old.log"
    fresh.write_text("[2026-07-24T12:00:00Z] stored=6 failed=0 ()\n")
    old.write_text("[2026-07-24T00:00:00Z] FAILED IOException: locked\n")
    now = int(__import__("datetime").datetime(
        2026, 7, 24, 12, 30, tzinfo=__import__("datetime").timezone.utc
    ).timestamp())
    out = mc.sidecar_health(now, {
        "fresh": (fresh, 3600),
        "old": (old, 3600),
        "absent": (tmp_path / "gone.log", 3600),
    })
    assert out["fresh"]["stale"] is False and out["fresh"]["failed"] is False
    assert out["old"]["stale"] is True          # 12.5h > 2.5x cadence
    assert out["old"]["failed"] is True
    assert out["absent"]["stale"] is True       # never ran cannot hide
    assert out["absent"]["age_s"] is None


def test_stale_sidecar_warns_but_never_masks_trading_health():
    base = dict(now=1_000_000, heartbeat_ts=1_000_000 - 60,
                task_status="Running", churn_events=0,
                latest_candle_ts=1_000_000 - 120, supervisor_seen=True)
    sidecars = {"backup": {"stale": True, "failed": False, "age_s": 99999,
                           "cadence_s": 21600, "last_line": ""}}
    tier, reasons = mc.classify_health(**base, sidecars=sidecars)
    assert tier == "WARNING"
    assert any("backup" in r for r in reasons)

    # A dead trader must still read CRITICAL, not be diluted to WARNING.
    dead = {**base, "heartbeat_ts": 1_000_000 - mc.HEARTBEAT_STALE_S - 1}
    tier2, _ = mc.classify_health(**dead, sidecars=sidecars)
    assert tier2 == "CRITICAL"


# ------------------------------------------------------- settle_missing (#7)

def test_index_closed_options_maps_every_id_in_the_list():
    payload = {"msg": {"closed_options": [
        {"id": [111, 112], "win": "win", "amount": 1.0, "win_amount": 1.87},
        {"id": [222], "win": "loose", "amount": 1.0, "win_amount": 0.0},
        {"id": [333], "win": "equal", "amount": 1.0, "win_amount": 1.0},
    ]}}
    out = index_closed_options(payload)
    assert out[111] == ("win", pytest.approx(0.87))
    assert out[112] == ("win", pytest.approx(0.87))   # list ids share outcome
    assert out[222] == ("loose", pytest.approx(-1.0))
    assert out[333] == ("equal", 0.0)


def test_index_closed_options_survives_garbage():
    assert index_closed_options({}) == {}
    assert index_closed_options({"msg": {}}) == {}
    assert index_closed_options({"msg": {"closed_options": None}}) == {}
    # one malformed entry must not lose the good ones
    payload = {"msg": {"closed_options": [
        {"no_win_key": 1},
        {"id": [9], "win": "win", "amount": 1.0, "win_amount": 1.9},
    ]}}
    assert list(index_closed_options(payload)) == [9]


# ---------------------------------------------------------- backup_state (#5)

def test_backup_isolates_per_file_failures(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "logs").mkdir(parents=True)
    (project / "logs" / "live_h2.jsonl").write_text('{"ts": 1}\n')
    (project / "logs" / "watchdog.log").write_text("[2026-07-24T00:00:00Z] x\n")
    dest = tmp_path / "backup"

    real_copy = backup_state._copy_atomic

    def flaky(src, dst):
        if src.name == "live_h2.jsonl":
            raise PermissionError("locked by writer")
        return real_copy(src, dst)

    monkeypatch.setattr(backup_state, "_copy_atomic", flaky)
    result = backup_state.run_backup(project, dest)

    # the locked file is reported, the others still got backed up
    assert any("live_h2.jsonl" in f for f in result["failed"])
    assert "logs/watchdog.log" in result["copied"]
    assert (dest / "logs_watchdog.log").exists()


def test_backup_leaves_no_temp_files_and_replaces_atomically(tmp_path):
    project = tmp_path / "proj"
    (project / "logs").mkdir(parents=True)
    src = project / "logs" / "live_h2.jsonl"
    src.write_text('{"ts": 1}\n')
    dest = tmp_path / "backup"

    backup_state.run_backup(project, dest)
    src.write_text('{"ts": 1}\n{"ts": 2}\n')
    backup_state.run_backup(project, dest)

    assert (dest / "logs_live_h2.jsonl").read_text().count("\n") == 2
    assert not list(dest.glob("*.tmp"))


def test_backup_reports_missing_by_name(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    result = backup_state.run_backup(project, tmp_path / "b")
    assert "journal.db" in result["missing"]
    assert result["copied"] == [] and result["failed"] == []
