"""Recover broker verdicts for orphaned orders - run by hand when needed.

If the runner dies between placing an order and settling it (reboot, task
kill, crash), live_h2.jsonl keeps the signal row with an order_id but never
receives its settled duplicate - and the label-fidelity measurement quietly
loses a trade.

IMPORTANT (audit 2026-07-24): this must NOT use check_win_v4. That call
blocks on api.socket_option_closed[id], a dict populated only by the
close event of an option bought in the SAME session - for an orphan from a
dead process the key never appears, so every recovery would hang until its
timeout. The broker's closed-options history (get_optioninfo_v2) is the
only channel that can answer for past orders, so we page through it and
match by id.

Run while the runner is idle if possible (both append to the same file).

Usage: .venv\\Scripts\\python.exe settle_missing.py [--dry-run] [--limit 200]
"""

import argparse
import json
import sys
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


def index_closed_options(payload) -> dict[int, tuple[str, float]]:
    """{order_id: (win, profit)} from a get_optioninfo_v2 response.

    The broker returns each closed option with 'id' as a LIST (see the
    vendored check_win_v3), so every id in it maps to the same outcome.
    Profit mirrors the runner's own convention: equal -> 0.0, otherwise
    win_amount - amount (a loss has win_amount 0, giving -amount).
    Pure - unit-tested."""
    out: dict[int, tuple[str, float]] = {}
    try:
        options = payload["msg"]["closed_options"]
    except (KeyError, TypeError):
        return out
    for opt in options or []:
        try:
            win = opt["win"]
            amount = float(opt.get("amount") or 0.0)
            win_amount = float(opt.get("win_amount") or 0.0)
            profit = 0.0 if win == "equal" else win_amount - amount
            ids = opt.get("id")
            ids = ids if isinstance(ids, (list, tuple)) else [ids]
            for oid in ids:
                if oid is not None:
                    out[int(oid)] = (win, profit)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=200,
                    help="closed options to page from the broker (default 200)")
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

    try:
        payload = _call(client.get_optioninfo_v2, args.limit, timeout=90)
    except Exception as exc:
        raise SystemExit(f"could not read closed-options history: "
                         f"{type(exc).__name__}: {exc}")
    verdicts = index_closed_options(payload)
    print(f"broker history: {len(verdicts)} closed option id(s)")

    recovered, unmatched = 0, []
    for record in orphans:
        oid = record["order_id"]
        found = verdicts.get(int(oid))
        if found is None:
            unmatched.append(oid)
            continue
        result, profit = found
        outcome = {**record, "result": result, "profit": profit,
                   "settled": True, "recovered_by": "settle_missing"}
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(outcome) + "\n")
        recovered += 1
        print(f"  recovered {oid}: {result} {profit}")

    print(f"recovered {recovered}/{len(orphans)}")
    if unmatched:
        print(f"  not in the last {args.limit} closed options: "
              f"{', '.join(str(o) for o in unmatched)}\n"
              f"  retry with a larger --limit", file=sys.stderr)
    # Nonzero when the job did not fully succeed, so a scripted caller can
    # tell "nothing to do" (0 orphans, exit 0) from "could not recover".
    return 0 if not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
