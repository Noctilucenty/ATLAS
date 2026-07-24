"""Evidence backup - offsite copies of the forward test's irreplaceable state.

Code and models live in git; market.duckdb is rebuildable from the broker
(~60 days). What is NOT recoverable if this disk dies: the paper/trade log
(live_h2.jsonl - the forward test's entire evidence), its heartbeats, the
trade journal, and the ops logs. This copies them every 6 h (Task
Scheduler: ATLAS-backup) into a OneDrive-synced folder - cloud-offsite for
free. Plain file copies of append-only files: no live-database locks, no
sync corruption (the reason the live system moved OUT of OneDrive).

Usage: python backup_state.py
"""

import shutil
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


def run_backup(project: Path = PROJECT_DIR, dest: Path = BACKUP_DIR) -> dict:
    """Copy existing targets; returns {copied, missing, bytes}. Pure enough
    to test with tmp dirs."""
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing, total = [], [], 0
    for rel in TARGETS:
        src = project / rel
        if not src.exists():
            missing.append(rel)
            continue
        out = dest / rel.replace("/", "_")
        shutil.copy2(src, out)
        copied.append(rel)
        total += src.stat().st_size
    return {"copied": copied, "missing": missing, "bytes": total}


def main() -> int:
    result = run_backup()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (f"[{stamp}] copied={len(result['copied'])} "
            f"missing={len(result['missing'])} bytes={result['bytes']:,}")
    log_path = PROJECT_DIR / "logs" / "backup.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
