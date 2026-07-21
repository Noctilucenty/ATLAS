"""Collection-only data-health and coverage report.

Reads DuckDB and the collector log ONLY - no broker login, no training, no
model or gate changes. Run at the end of every collection cycle (and any
time by hand) to monitor exactly what the reviewer asked for: cycle exit
statuses, gaps, conflicts, required payout-key presence, distinct UTC days,
and the prospective payout coverage estimate that gates the next research
review.

Usage: python health_report.py     (prints JSON, writes logs/health.json)
"""

import bisect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOG_PATH = PROJECT_DIR / "logs" / "collector.log"
HEALTH_PATH = PROJECT_DIR / "logs" / "health.json"

PAYOUT_MAX_AGE_S = 7200   # must match train.py --payout-max-age default
COVERAGE_WINDOW_H = 48


def parse_cycle_statuses(log_text: str, last: int = 24) -> list[int]:
    """Exit statuses of the most recent collection cycles, oldest first."""
    statuses = [int(m) for m in re.findall(r"=== cycle exit status: (\d+) ===", log_text)]
    return statuses[-last:]


def payout_coverage_estimate(
    candle_to_ts: list[int], snapshot_ts: list[int], max_age_s: int = PAYOUT_MAX_AGE_S
) -> float | None:
    """Share of candle close-times with a causal snapshot within max age.

    Mirrors storage.latest_payout_before semantics: only snapshots at or
    before the timestamp count, and older-than-max-age is unavailable."""
    if not candle_to_ts:
        return None
    snapshot_ts = sorted(snapshot_ts)
    covered = 0
    for ts in candle_to_ts:
        idx = bisect.bisect_right(snapshot_ts, ts)
        if idx and ts - snapshot_ts[idx - 1] <= max_age_s:
            covered += 1
    return covered / len(candle_to_ts)


def build_report(conn, log_text: str, now_epoch: int) -> dict:
    from instruments import INSTRUMENTS
    from storage import load_canonical_history

    statuses = parse_cycle_statuses(log_text)
    report: dict = {
        "generated_utc": datetime.fromtimestamp(now_epoch, timezone.utc).isoformat(),
        "cycles": {
            "recent_exit_statuses": statuses,
            "recent_failures": sum(1 for s in statuses if s != 0),
        },
        "instruments": {},
        "healthy": True,
    }

    latest_batch = conn.execute("SELECT max(ts_epoch) FROM payout_snapshots").fetchone()[0]
    window_start = now_epoch - COVERAGE_WINDOW_H * 3600

    for asset, spec in INSTRUMENTS.items():
        history, hist_report = load_canonical_history(conn, spec.candle_asset, 60)
        snapshot_ts = [
            r[0]
            for r in conn.execute(
                "SELECT ts_epoch FROM payout_snapshots WHERE asset = ? AND kind = ? ORDER BY ts_epoch",
                (spec.quote_key, spec.option_kind),
            ).fetchall()
        ]
        if history.empty:
            entry = {"candles": 0, "error": "no data"}
            report["healthy"] = False
        else:
            recent_to_ts = [int(t) for t in history["to_ts"] if t >= window_start]
            days = {
                datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")
                for t in history["from_ts"]
            }
            in_latest_batch = bool(
                latest_batch is not None
                and conn.execute(
                    "SELECT count(*) FROM payout_snapshots WHERE ts_epoch = ? AND asset = ? AND kind = ?",
                    (latest_batch, spec.quote_key, spec.option_kind),
                ).fetchone()[0]
            )
            entry = {
                "candles": hist_report["candles"],
                "datasets": len(hist_report["datasets_used"]),
                "gaps": len(hist_report["gaps"]),
                "conflicts": len(hist_report["conflicts"]),
                "distinct_utc_days": len(days),
                "last_candle_age_s": now_epoch - int(history["to_ts"].max()),
                "payout_snapshots": len(snapshot_ts),
                "payout_key_in_latest_batch": in_latest_batch,
                f"prospective_coverage_{COVERAGE_WINDOW_H}h": payout_coverage_estimate(
                    recent_to_ts, snapshot_ts
                ),
            }
            if hist_report["conflicts"] or not in_latest_batch:
                report["healthy"] = False
        report["instruments"][asset] = entry

    if report["cycles"]["recent_exit_statuses"] and report["cycles"]["recent_exit_statuses"][-1] != 0:
        report["healthy"] = False
    return report


def main() -> int:
    from storage import open_db

    conn = open_db()
    log_text = LOG_PATH.read_text() if LOG_PATH.exists() else ""
    report = build_report(conn, log_text, int(datetime.now(timezone.utc).timestamp()))
    HEALTH_PATH.parent.mkdir(exist_ok=True)
    HEALTH_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
