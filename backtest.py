"""Run MIDAS binary backtests over one immutable run bundle.

A run bundle (written atomically by train.py) contains candles.json,
signals.json, folds.json and manifest.json whose SHA-256 hashes bind them
together. Before invoking MIDAS the bundle is VERIFIED: if either artifact's
hash differs from the manifest, the backtest aborts - new signals can never
be silently replayed against stale candles or vice versa.

MIDAS is invoked once per walk-forward fold and once in aggregate; per-fold
verdicts expose instability that an aggregate can hide. --payout-prospective
is passed to MIDAS only when the manifest declares prospective payouts AND
every evaluated row had a causally valid snapshot (payout_coverage == 1.0).

Usage:
  python backtest.py [runs/EURUSD/<run-id>]   (default: newest bundle)

MIDAS binary location: $MIDAS_BIN, falling back to the release build in the
sibling MIDAS repo (cargo build --release --bin binary-backtest).
"""

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"
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


def latest_run_dir(root: Path = RUNS_DIR) -> Path:
    candidates = (
        [
            d
            for asset_dir in root.iterdir()
            if asset_dir.is_dir()
            for d in asset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".tmp") and (d / "manifest.json").exists()
        ]
        if root.exists()
        else []
    )
    if not candidates:
        raise SystemExit(f"no run bundles under {root} - run train.py first")
    return max(candidates, key=lambda d: d.name)


def verify_bundle(run_dir: Path) -> dict:
    """Check every artifact against the manifest hashes; abort on mismatch.

    Covers candles, signals, folds AND the MIDAS binary that will execute the
    replay - if the engine changed since the bundle was written, its replay
    semantics may differ and the bundle must be regenerated, not reused."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{run_dir} has no manifest.json - not a run bundle")
    manifest = json.loads(manifest_path.read_text())
    artifacts = (
        ("candles.json", "candles_sha256"),
        ("signals.json", "signals_sha256"),
        ("folds.json", "folds_sha256"),
    )
    for name, key in artifacts:
        path = run_dir / name
        if not path.exists():
            raise SystemExit(f"ABORT: {run_dir}/{name} is missing")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest.get(key)
        if actual != expected:
            raise SystemExit(
                f"ABORT: {name} hash {actual[:16]} != manifest {str(expected)[:16]} - "
                "stale or tampered bundle; regenerate with train.py"
            )

    expected_binary = manifest.get("midas_binary_sha256")
    actual_binary = hashlib.sha256(midas_bin().read_bytes()).hexdigest()
    if actual_binary != expected_binary:
        raise SystemExit(
            f"ABORT: MIDAS binary hash {actual_binary[:16]} != bundled "
            f"{str(expected_binary)[:16]} - the engine changed since this bundle "
            "was written; regenerate with train.py"
        )
    return manifest


def run_midas(candles: Path, signals: Path, bankroll: float, extra: list[str]) -> dict:
    result = subprocess.run(
        [str(midas_bin()), str(candles), str(signals), "--bankroll", str(bankroll), *extra],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"MIDAS failed on {signals.name}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def fold_key(signal: dict) -> str:
    note = signal.get("note") or "unknown"
    return note.split(",")[0]


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run_dir()
    manifest = verify_bundle(run_dir)
    bankroll = float(os.environ.get("BANKROLL", "100"))

    prospective_ok = (
        manifest.get("payout_source") == "prospective"
        and manifest.get("payout_coverage") == 1.0
    )
    gate_flags = ["--payout-prospective"] if prospective_ok else []

    candles = run_dir / "candles.json"
    signals = json.loads((run_dir / "signals.json").read_text())

    out_dir = run_dir / "midas"
    out_dir.mkdir(exist_ok=True)

    by_fold: dict[str, list[dict]] = defaultdict(list)
    for signal in signals:
        by_fold[fold_key(signal)].append(signal)

    summary: dict = {
        "run_dir": str(run_dir),
        "dataset_content_sha256": manifest.get("dataset_content_sha256"),
        "payout_source": manifest.get("payout_source"),
        "payout_coverage": manifest.get("payout_coverage"),
        "payout_prospective_flag": prospective_ok,
        "bankroll": bankroll,
        "folds": {},
    }
    for fold, fold_signals in sorted(by_fold.items()):
        fold_file = out_dir / f"signals_{fold.replace('=', '_')}.json"
        fold_file.write_text(json.dumps(fold_signals))
        report = run_midas(candles, fold_file, bankroll, gate_flags)
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

    aggregate = run_midas(candles, run_dir / "signals.json", bankroll, gate_flags)
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
    main()
