"""Portable always-on supervisor - one process replacing launchd + the zsh
scripts. Runs identically on Windows, macOS and Linux.

Two responsibilities, forever:
  COLLECT  once at start and every ~60 min: fetch candles for every
           registered instrument + a payout snapshot (what collect_cycle.sh
           did, in Python).
  TRADE    keep a single live_h2_runner session alive continuously,
           restarting it whenever it exits (the 57-min reconnect design, or
           a crash bail). The runner's own socket lock guarantees exactly
           one trader even if this supervisor is accidentally double-run.

Launch it with the platform's startup mechanism and never think about it:
  Windows  Task Scheduler at logon, "restart on failure" (see WINDOWS_SETUP.md)
  macOS    launchd, or just `caffeinate -i python supervisor.py`
  Linux    systemd, or nohup

Flags:
  --paper        run the trader WITHOUT --trade (log signals, place nothing)
  --collect-min  minutes between collection cycles (default 60)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from instruments import INSTRUMENTS

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable  # the venv's interpreter, whatever the OS
LOG = PROJECT_DIR / "logs"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    LOG.mkdir(exist_ok=True)
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with open(LOG / "supervisor.log", "a") as fh:
        fh.write(line + "\n")


def collect_cycle() -> None:
    """Candles for all instruments + one payout snapshot. Errors are logged,
    never fatal - a bad cycle must not stop the supervisor."""
    assets = list(INSTRUMENTS)
    try:
        r = subprocess.run(
            [PYTHON, "collector.py", "candles", *assets, "--interval", "60", "--hours", "2"],
            cwd=PROJECT_DIR, capture_output=True, text=True, timeout=1200,
        )
        log(f"collect candles exit={r.returncode}"
            + ("" if r.returncode == 0 else f" stderr={r.stderr[-200:]}"))
        r2 = subprocess.run(
            [PYTHON, "collector.py", "payouts"],
            cwd=PROJECT_DIR, capture_output=True, text=True, timeout=300,
        )
        log(f"collect payouts exit={r2.returncode}")
    except Exception as exc:
        log(f"collect cycle error: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", action="store_true",
                    help="run the trader without --trade (log only, place nothing)")
    ap.add_argument("--collect-min", type=int, default=60)
    args = ap.parse_args()

    log(f"supervisor start (python={PYTHON}, trade={'off' if args.paper else 'on'})")
    last_collect = 0.0
    runner = None
    runner_cmd = [PYTHON, "live_h2_runner.py", "--minutes", "57"]
    if not args.paper:
        runner_cmd.append("--trade")

    try:
        while True:
            now = time.time()
            # Collection on schedule.
            if now - last_collect >= args.collect_min * 60:
                collect_cycle()
                last_collect = now
            # Keep exactly one trader alive.
            if runner is None or runner.poll() is not None:
                if runner is not None:
                    log(f"runner exited ({runner.returncode}); relaunching")
                runner = subprocess.Popen(runner_cmd, cwd=PROJECT_DIR)
                log(f"runner launched pid={runner.pid}")
            time.sleep(30)
    except KeyboardInterrupt:
        log("supervisor stopping (KeyboardInterrupt)")
    finally:
        if runner is not None and runner.poll() is None:
            runner.terminate()
            try:
                runner.wait(timeout=30)
            except subprocess.TimeoutExpired:
                runner.kill()
        log("supervisor stopped")


if __name__ == "__main__":
    main()
