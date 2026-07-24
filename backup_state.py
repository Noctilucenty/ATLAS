"""Evidence backup - offsite copies of the forward test's irreplaceable state.

Code and models live in git; market.duckdb is rebuildable from the broker
(~60 days). What is NOT recoverable if this disk dies: the paper/trade log
(live_h2.jsonl - the forward test's entire evidence), its heartbeats, the
trade journal, and the ops logs. This copies them every 6 h (Task
Scheduler: ATLAS-backup) into a OneDrive-synced folder - cloud-offsite for
free. Plain file copies of append-only files: no live-database locks, no
sync corruption (the reason the live system moved OUT of OneDrive).

Robustness (audit 2026-07-24): each file is copied independently, so one
locked or vanished file cannot abort the rest, and each copy lands via a
.tmp sidecar + atomic replace, so an interrupted run can never truncate
the previous good backup. journal.db uses sqlite's own backup API when
possible (a plain copy of a live sqlite file can be torn).

Usage: python backup_state.py
"""

import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BACKUP_DIR = Path.home() / "OneDrive" / "Desktop" / "ATLAS-evidence-backup"

# Irreplaceable, append-only or rarely-written state.
TARGETS = [
    "logs/live_h2.jsonl",
    "logs/live_h2_heartbeat.jsonl",
    "logs/supervisor.log",
    "logs/task_run.log",
    "logs/extra_collect.log",
    "logs/catchup.log",
    "logs/watchdog.log",
    "logs/payout_landscape.json",
    "logs/universe_profile.json",
    "journal.db",
    ".env",  # credentials are part of disaster recovery; folder is the
             # user's private OneDrive, same trust domain as the original
]


def _copy_atomic(src: Path, dst: Path) -> None:
    """Copy via a temp sidecar then atomically replace, so a crash or a
    locked destination never leaves a half-written backup in place."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _copy_sqlite(src: Path, dst: Path) -> None:
    """Consistent copy of a live sqlite file via its backup API, then the
    same atomic replace. Falls back to a plain copy if the file is not a
    readable database."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
        try:
            target = sqlite3.connect(str(tmp))
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
    except sqlite3.Error:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def run_backup(project: Path = PROJECT_DIR, dest: Path = BACKUP_DIR) -> dict:
    """Copy existing targets independently.

    Returns {copied, missing, failed, bytes}: 'missing' is benign (a log
    that does not exist yet), 'failed' is not - it means a file that IS
    there could not be backed up."""
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing, failed, total = [], [], [], 0
    for rel in TARGETS:
        src = project / rel
        try:
            if not src.exists():
                missing.append(rel)
                continue
            out = dest / rel.replace("/", "_")
            if src.suffix == ".db":
                _copy_sqlite(src, out)
            else:
                _copy_atomic(src, out)
            copied.append(rel)
            total += src.stat().st_size
        except Exception as exc:
            failed.append(f"{rel} ({type(exc).__name__})")
    return {"copied": copied, "missing": missing, "failed": failed,
            "bytes": total}


def main() -> int:
    log_path = PROJECT_DIR / "logs" / "backup.log"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        result = run_backup()
    except Exception as exc:  # dest unwritable, OneDrive gone, disk full
        line = f"[{stamp}] FAILED {type(exc).__name__}: {exc}"
        result = None
    else:
        line = (f"[{stamp}] copied={len(result['copied'])} "
                f"bytes={result['bytes']:,}"
                + (f" MISSING={','.join(result['missing'])}"
                   if result["missing"] else "")
                + (f" FAILED={','.join(result['failed'])}"
                   if result["failed"] else ""))
    try:
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # losing the log line must not also lose the backup's exit code
    print(line)
    # Nonzero when the backup did not do its job, so Task Scheduler's Last
    # Result and the sidecar health check can both see it.
    if result is None or result["failed"] or not result["copied"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
