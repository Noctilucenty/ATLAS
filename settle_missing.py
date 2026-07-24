"""Recover broker verdicts for orphaned orders - run by hand when needed.

If the runner dies between placing an order and settling it (reboot, task
kill, crash), live_h2.jsonl keeps the signal row with an order_id but never
receives its settled duplicate - and the label-fidelity measurement quietly
loses a trade. This tool finds those orphans, asks the broker for their
(long-settled) verdicts via check_win_v4, and appends the missing settled
rows.

Run it while the runner is idle if possible (both append to the same file;
appends are small and line-buffered, so interleaving is unlikely but not
worth inviting).

Usage: .venv\\Scripts\\python.exe settle_missing.py [--dry-run]
"""

import argparse
import json
import sys
import threading
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOG_PATH = PROJECT_DIR / "logs" / "live_h2.jsonl"


def find_unsettled(rows: list[dict]) -> list[dict]:
    """Signal rows whose order_id never received a settled duplicate.
    Pure - unit-tested."""
    settled_ids = {r.get("order_id") for r in rows if r.get("settled")}
    return [r for r in rows
            if r.get("order_id") and not r.get("settled")
            and r["order_id"] not in settled_ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    orphans = find_unsettled(rows)
    print(f"{len(orphans)} unsettled order(s)")
    if not orphans or args.dry_run:
        return 0

    import os

    from run_once import _call, _load_env

    _load_env()
    from iqoptionapi.stable_api import IQ_Option

    client = IQ_Option(os.environ["IQ_EMAIL"], os.environ["IQ_PASSWORD"])
    ok, reason = _call(client.connect, timeout=90)
    if not ok:
        raise SystemExit(f"login failed: {reason}")

    recovered = 0
    for record in orphans:
        try:
            result, profit = _call(client.check_win_v4, record["order_id"],
                                   timeout=60)
            outcome = {**record, "result": result, "profit": profit,
                       "settled": True, "recovered_by": "settle_missing"}
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(outcome) + "\n")
            recovered += 1
            print(f"  recovered {record['order_id']}: {result} {profit}")
        except Exception as exc:
            print(f"  WARN {record['order_id']}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    print(f"recovered {recovered}/{len(orphans)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
