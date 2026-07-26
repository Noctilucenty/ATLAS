"""One-command execution probe - correct sequencing, no coordination mistakes.

The fidelity measurement needs three things to happen in order: the quote
recorder must be running BEFORE any order expires (otherwise there is no
sub-minute stream and the settlement-rule attribution is impossible), the
orders must be placed, and settlement must be collected AFTER the last expiry.
Doing that by hand across three shells invites exactly one mistake - starting
the recorder late - which silently costs the attribution while still producing
a plausible-looking result.

This orchestrates all three. It shells out to the existing tools rather than
reimplementing them, so the order-placing code path is untouched and still
carries its own PRACTICE guards and --confirm requirement.

Timeline for N trades spaced S seconds apart with a 15-minute expiry:
  recorder covers   now .. (N-1)*S + 15min + margin
  placement takes   (N-1)*S
  last expiry at    placement end + 15min

Usage:
  .venv\\Scripts\\python.exe probe_run.py --dry-run
  .venv\\Scripts\\python.exe probe_run.py --trades 6 --confirm
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
EXPIRY_MINUTES = 15
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)


def plan(trades: int, spacing: int, margin_s: int = 180) -> dict:
    """Timeline in seconds from now. Pure - unit-tested."""
    placement_s = max(0, (trades - 1) * spacing)
    last_expiry_s = placement_s + EXPIRY_MINUTES * 60
    return {
        "placement_s": placement_s,
        "last_expiry_s": last_expiry_s,
        # The recorder must outlive the final expiry, or the trimmed-average
        # label for the last trade cannot be computed.
        "recorder_s": last_expiry_s + margin_s,
        "settle_wait_s": last_expiry_s + 60,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="EURUSD")
    ap.add_argument("--trades", type=int, default=6)
    ap.add_argument("--spacing", type=int, default=120)
    ap.add_argument("--direction", choices=("call", "put", "alternate"),
                    default="alternate")
    ap.add_argument("--confirm", action="store_true",
                    help="REQUIRED to place orders")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t = plan(args.trades, args.spacing)
    mins = lambda s: f"{s / 60:.1f} min"  # noqa: E731
    print(f"=== execution probe plan: {args.trades} x $1 PRACTICE "
          f"{EXPIRY_MINUTES}m binaries on {args.asset} ===")
    print(f"  quote recorder runs   {mins(t['recorder_s'])}")
    print(f"  placement spans       {mins(t['placement_s'])} "
          f"({args.spacing}s apart, direction={args.direction})")
    print(f"  last expiry at        +{mins(t['last_expiry_s'])}")
    print(f"  settlement collected  +{mins(t['settle_wait_s'])}")
    print(f"  total runtime         ~{mins(t['settle_wait_s'] + 60)}")

    if args.dry_run or not args.confirm:
        if not args.dry_run:
            print("\nrefusing to place orders without --confirm", file=sys.stderr)
            return 2
        print("\n(dry run - nothing started)")
        return 0

    # 1. Recorder FIRST, detached, covering the whole window.
    until = int(time.time() + t["recorder_s"])
    rec = subprocess.Popen(
        [str(PYTHON), "quote_recorder.py", "--asset", args.asset,
         "--until-ts", str(until), "--interval", "2"],
        cwd=PROJECT_DIR, creationflags=NO_WINDOW | DETACHED,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL)
    print(f"\n[1/3] quote recorder started (pid {rec.pid}), until "
          f"{datetime.fromtimestamp(until, timezone.utc):%H:%M:%S}Z", flush=True)
    time.sleep(5)  # let a few quotes land before the first order

    try:
        # 2. Place the orders (its own PRACTICE guards apply).
        print(f"[2/3] placing {args.trades} order(s)...", flush=True)
        place = subprocess.run(
            [str(PYTHON), "probe_execution.py", "--asset", args.asset,
             "--trades", str(args.trades), "--spacing", str(args.spacing),
             "--direction", args.direction, "--confirm"],
            cwd=PROJECT_DIR, creationflags=NO_WINDOW, text=True)
        if place.returncode != 0:
            print(f"placement exited {place.returncode}", file=sys.stderr)

        # 3. Wait out the final expiry, then settle.
        wait = max(0, t["settle_wait_s"] - t["placement_s"])
        print(f"[3/3] waiting {mins(wait)} for the last expiry...", flush=True)
        time.sleep(wait)
        subprocess.run(
            [str(PYTHON), "probe_execution.py", "--asset", args.asset,
             "--settle-only"],
            cwd=PROJECT_DIR, creationflags=NO_WINDOW, text=True)
    finally:
        # The recorder exits on its own deadline, but never leave it running
        # if we bailed early.
        if rec.poll() is None:
            rec.terminate()
            print("quote recorder stopped", flush=True)

    print("\nprobe complete. Raw records: logs/execution_probe.jsonl")
    print("Strike detail (if a fill exists): "
          ".venv\\Scripts\\python.exe probe_order_record.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
