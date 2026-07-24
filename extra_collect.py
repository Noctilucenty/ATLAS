"""Sidecar collector for EXTRA instruments outside the frozen verdict
universe (RESEARCH_QUEUE.md item 2 - data-only, pre-verdict legal).

Runs hourly from Task Scheduler (task: ATLAS-extra-collect, pythonw, no
window), offset from the supervisor's collection cycle to avoid DuckDB
write-lock collisions. Deliberately does NOT touch instruments.py, the
supervisor, or the runner: these assets produce no signals and no trades
until they are registered as their own hypothesis after the current
forward window.

The vendored iqoptionapi constants predate these actives, so their
server-confirmed ids (probed live 2026-07-24 via update_ACTIVES_OPCODE)
are injected into the ACTIVES map at runtime - the vendor tree stays
pristine and collector.py stays untouched.

Failures are normal: exchange-hours assets (USSPX500, US30, ...) return
no candles while closed; the next open-hours run backfills (broker keeps
~60 days). Payouts need no sidecar - the supervisor's hourly snapshot
already captures every quoted key.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOG = PROJECT_DIR / "logs" / "extra_collect.log"

# Server-confirmed active ids missing from the vendored constants.
EXTRA_ACTIVE_IDS = {
    "SpaceX-OTC": 2443,   # broker-synthetic 24/7 (OTC-like pricing)
    "SpaceX-op": 2444,    # broker-synthetic, quoted hours
    "SP500-OTC": 1971,    # broker-synthetic S&P variant
    # Already in vendored constants, listed for completeness/collection:
    "USSPX500": 1239,     # real S&P 500 feed, US hours
    "US30": 1235,         # Dow
    "USNDAQ100": 1236,    # Nasdaq 100
    "UK100": 1241,        # FTSE 100
}
LOOKBACK_HOURS = 4.0  # self-heals a missed cycle


def main() -> int:
    from iqoptionapi import constants

    for name, active_id in EXTRA_ACTIVE_IDS.items():
        constants.ACTIVES.setdefault(name, active_id)

    from collector import collect_candles

    results = collect_candles(list(EXTRA_ACTIVE_IDS), 60, LOOKBACK_HOURS)
    stored = [r for r in results if "dataset_id" in r]
    failed = [r for r in results if "error" in r]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ok_part = ", ".join(f"{r['asset']}:{r['candles']}" for r in stored)
    bad_part = ", ".join(r["asset"] for r in failed)
    line = (f"[{stamp}] stored={len(stored)} ({ok_part}) "
            f"failed={len(failed)} ({bad_part})")
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(json.dumps(results, indent=2))
    # Exit 0 always: closed-market cycles are expected, and Task Scheduler
    # must not treat them as task failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
