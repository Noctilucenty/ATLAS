"""Research experiment registry - the honest denominator.

Every screening experiment ATLAS runs is a lottery ticket: the more variants
tried, the higher the bar any "winner" must clear (see
validation_stats.deflated_win_rate). This registry exists so that N is a
recorded fact, not a flattering guess.

Append-only JSONL (research_registry.jsonl). Each entry is one experiment
FAMILY with the number of variants it evaluated - e.g. a 5x6 threshold grid
is one entry with n_variants=30. total_trials() is the deflation denominator.

Distinct from experiments.py (train.py's per-run provenance journal): this
counts research attempts across ALL tooling, including grids that never
produced a run bundle.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "research_registry.jsonl"


def record(family: str, description: str, n_variants: int,
           config: dict | None = None, outcome: str = "") -> dict:
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "family": family,
        "description": description,
        "n_variants": int(n_variants),
        "config_hash": hashlib.sha256(
            json.dumps(config or {}, sort_keys=True).encode()
        ).hexdigest()[:12],
        "config": config or {},
        "outcome": outcome,
    }
    with open(REGISTRY, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def entries() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]


def total_trials() -> int:
    """The deflation denominator: every variant ever evaluated, plus a floor
    of 1 so an empty registry cannot flatter anyone."""
    return max(sum(e["n_variants"] for e in entries()), 1)
