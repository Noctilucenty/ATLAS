"""Append-only experiment ledger.

Every training run - kept or rejected - appends one JSON line. Nothing is
ever rewritten, so the ledger is the honest record of how many variants were
attempted against a dataset (the denominator backtest-overfitting math needs).
Each entry pins full provenance: dataset content hash, source commit, feature
code hash, dependency lock hash, complete parameters, fold time ranges and
payout source.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LEDGER_PATH = PROJECT_DIR / "experiments.jsonl"
LOCK_PATH = PROJECT_DIR / "requirements.lock"


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _source_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return None


def record_experiment(
    *,
    dataset_content_sha256: str | None,
    parameters: dict,
    fold_ranges: list[dict],
    payout_source: str,
    outcome: str,
    ledger_path: Path = LEDGER_PATH,
) -> dict:
    """Append one experiment entry and return it (with its 1-based id)."""
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_content_sha256": dataset_content_sha256,
        "source_commit": _source_commit(),
        "feature_code_hash": _sha256_file(PROJECT_DIR / "features.py"),
        "train_code_hash": _sha256_file(PROJECT_DIR / "train.py"),
        "lock_hash": _sha256_file(LOCK_PATH),
        "parameters": parameters,
        "fold_ranges": fold_ranges,
        "payout_source": payout_source,
        "outcome": outcome,
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    entry["id"] = sum(1 for _ in open(ledger_path))
    return entry


def count_variants(
    dataset_content_sha256: str | None, ledger_path: Path = LEDGER_PATH
) -> int:
    """How many experiments have run against this exact dataset."""
    if not ledger_path.exists():
        return 0
    count = 0
    for line in ledger_path.read_text().splitlines():
        if line.strip() and json.loads(line).get("dataset_content_sha256") == dataset_content_sha256:
            count += 1
    return count
