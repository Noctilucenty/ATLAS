"""Run MIDAS binary backtests per walk-forward fold, then in aggregate.

Reads the signals_out/ artifacts train.py produced, splits signals by their
fold tag, invokes the MIDAS `binary-backtest` CLI once per fold and once for
the full stream, and writes all reports to signals_out/midas/. Per-fold
verdicts expose instability that an aggregate can hide.

MIDAS binary location: $MIDAS_BIN, falling back to the release build in the
sibling MIDAS repo (cargo build --release --bin binary-backtest).
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SIGNALS_DIR = PROJECT_DIR / "signals_out"
DEFAULT_MIDAS_BIN = (
    PROJECT_DIR.parent / "MIDAS" / "target" / "release" / "binary-backtest"
)


def midas_bin() -> Path:
    path = Path(os.environ.get("MIDAS_BIN", DEFAULT_MIDAS_BIN))
    if not path.exists():
        raise SystemExit(
            f"MIDAS binary not found at {path} - build it with "
            "`cargo build --release --bin binary-backtest` or set MIDAS_BIN"
        )
    return path


def run_midas(candles: Path, signals: Path, bankroll: float, extra: list[str]) -> dict:
    result = subprocess.run(
        [str(midas_bin()), str(candles), str(signals), "--bankroll", str(bankroll), *extra],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"MIDAS failed on {signals.name}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def main() -> None:
    bankroll = float(os.environ.get("BANKROLL", "100"))
    candles = SIGNALS_DIR / "candles.json"
    signals_file = SIGNALS_DIR / "signals.json"
    if not candles.exists() or not signals_file.exists():
        raise SystemExit("run train.py first (missing signals_out artifacts)")

    signals = json.loads(signals_file.read_text())
    out_dir = SIGNALS_DIR / "midas"
    out_dir.mkdir(exist_ok=True)

    by_fold: dict[str, list[dict]] = defaultdict(list)
    for signal in signals:
        by_fold[signal.get("note") or "unknown"].append(signal)

    summary: dict = {"folds": {}, "bankroll": bankroll}
    for fold, fold_signals in sorted(by_fold.items()):
        fold_file = out_dir / f"signals_{fold.replace('=', '_')}.json"
        fold_file.write_text(json.dumps(fold_signals))
        report = run_midas(candles, fold_file, bankroll, [])
        (out_dir / f"report_{fold.replace('=', '_')}.json").write_text(
            json.dumps(report, indent=1)
        )
        metrics = report["report"]["metrics"]
        summary["folds"][fold] = {
            "executed": report["report"]["executed"],
            "net_pnl": metrics["net_pnl"],
            "win_rate": metrics["win_rate"],
            "brier": metrics["brier_score"],
        }

    aggregate = run_midas(candles, signals_file, bankroll, [])
    (out_dir / "report_aggregate.json").write_text(json.dumps(aggregate, indent=1))
    summary["aggregate"] = {
        "executed": aggregate["report"]["executed"],
        "net_pnl": aggregate["report"]["metrics"]["net_pnl"],
        "promotable": aggregate["verdict"]["promotable"],
        "failed_gates": [
            c["name"] for c in aggregate["verdict"]["checks"] if not c["passed"]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
