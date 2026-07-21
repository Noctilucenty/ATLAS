"""Tests for the promotion-readiness corrections: causal payout joins,
immutable run bundles with hash verification, and campaign accounting."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import verify_bundle  # noqa: E402
from features import FEATURE_COLUMNS  # noqa: E402
from storage import latest_payout_before, open_db  # noqa: E402
from train import walk_forward, write_run_bundle  # noqa: E402


# ---------------- causal payout join ----------------

def snapshot(conn, ts_epoch: int, payout: float, quote_key: str = "EURUSD-op"):
    conn.execute(
        """INSERT INTO payout_snapshots (ts_utc, ts_epoch, asset, kind, payout, source)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.fromtimestamp(ts_epoch, timezone.utc), ts_epoch, quote_key, "turbo", payout, "test"),
    )

def test_join_uses_latest_snapshot_at_or_before():
    conn = open_db(Path(":memory:"))
    snapshot(conn, 1000, 0.80)
    snapshot(conn, 2000, 0.85)
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 2500, 3600) == 0.85
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 2000, 3600) == 0.85  # at == allowed
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 1500, 3600) == 0.80

def test_join_never_uses_a_later_snapshot():
    conn = open_db(Path(":memory:"))
    snapshot(conn, 5000, 0.90)
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 4999, 999_999) is None

def test_join_rejects_stale_snapshots():
    conn = open_db(Path(":memory:"))
    snapshot(conn, 1000, 0.85)
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 1000 + 7200, 7200) == 0.85
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 1000 + 7201, 7200) is None

def test_join_is_instrument_and_kind_scoped():
    conn = open_db(Path(":memory:"))
    snapshot(conn, 1000, 0.86, quote_key="EURUSD-OTC")
    assert latest_payout_before(conn, "EURUSD-op", "turbo", 2000, 7200) is None
    assert latest_payout_before(conn, "EURUSD-OTC", "binary", 2000, 7200) is None
    assert latest_payout_before(conn, "EURUSD-OTC", "turbo", 2000, 7200) == 0.86


# ---------------- prospective walk-forward ----------------

def synthetic_features(n: int = 900, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame.insert(0, "from_ts", 1_000_000 + np.arange(n) * 60)
    frame.insert(1, "to_ts", frame["from_ts"] + 60)
    prob = 1 / (1 + np.exp(-2.5 * frame["rsi"]))
    frame["label_up"] = (rng.random(n) < prob).astype(float)
    frame["feature_version"] = "test"
    return frame

def test_prospective_mode_forces_no_trade_when_payout_missing():
    cutoff = 1_000_000 + 700 * 60
    lookup = lambda ts: 0.85 if ts <= cutoff else None  # noqa: E731
    result = walk_forward(synthetic_features(), payout=lookup, n_splits=3)
    missing = [s for s in result["signals"] if "payout_unavailable" in s["note"]]
    assert missing, "expected some rows past the snapshot cutoff"
    assert all(s["action"] == "no_trade" for s in missing)
    assert all(s["payout"] == 0.0 for s in missing)
    covered = [s for s in result["signals"] if "payout_unavailable" not in s["note"]]
    assert all(s["payout"] == 0.85 for s in covered)
    manifest = result["manifest"]
    assert manifest["payout_source"] == "prospective"
    assert 0 < manifest["payout_coverage"] < 1

def test_full_prospective_coverage_reports_one():
    result = walk_forward(synthetic_features(), payout=lambda ts: 0.85, n_splits=3)
    assert result["manifest"]["payout_coverage"] == 1.0
    assert result["manifest"]["assumed_payout"] is None


# ---------------- run bundles ----------------

def make_result() -> dict:
    return {
        "signals": [{"timestamp": "2026-07-21T00:00:00Z", "action": "no_trade", "note": "fold=0"}],
        "folds": [{"fold": 0}],
        "manifest": {"experiment_id": 7, "payout_source": "assumed"},
    }

def test_bundle_roundtrip_verifies(tmp_path):
    candles = [{"timestamp": "2026-07-21T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}]
    run_dir = write_run_bundle(tmp_path, "EURUSD", candles, make_result())
    assert run_dir.exists() and not run_dir.name.startswith(".tmp")
    manifest = verify_bundle(run_dir)  # must not raise
    assert manifest["experiment_id"] == 7
    assert manifest["candles_sha256"] and manifest["signals_sha256"]

def test_tampered_candles_abort_backtest(tmp_path):
    candles = [{"timestamp": "2026-07-21T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}]
    run_dir = write_run_bundle(tmp_path, "EURUSD", candles, make_result())
    stale = json.loads((run_dir / "candles.json").read_text())
    stale[0]["close"] = 9.9
    (run_dir / "candles.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="stale or tampered"):
        verify_bundle(run_dir)

def test_tampered_signals_abort_backtest(tmp_path):
    candles = [{"timestamp": "2026-07-21T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}]
    run_dir = write_run_bundle(tmp_path, "EURUSD", candles, make_result())
    (run_dir / "signals.json").write_text("[]")
    with pytest.raises(SystemExit, match="stale or tampered"):
        verify_bundle(run_dir)

def test_no_tmp_dir_left_behind(tmp_path):
    candles = [{"timestamp": "2026-07-21T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}]
    write_run_bundle(tmp_path, "EURUSD", candles, make_result())
    leftovers = [p for p in (tmp_path / "EURUSD").iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []


# ---------------- campaign accounting ----------------

def test_variants_counted_per_campaign_across_dataset_hashes(tmp_path):
    from experiments import count_variants, record_experiment

    ledger = tmp_path / "ledger.jsonl"
    for i, dataset_hash in enumerate(("hash-a", "hash-b", "hash-c")):
        record_experiment(
            campaign="eurusd-logreg",
            dataset_content_sha256=dataset_hash,
            parameters={"try": i},
            fold_ranges=[],
            payout_source="assumed",
            outcome="completed",
            ledger_path=ledger,
        )
    record_experiment(
        campaign="eurusd-otc-logreg",
        dataset_content_sha256="hash-a",
        parameters={},
        fold_ranges=[],
        payout_source="assumed",
        outcome="rejected",
        ledger_path=ledger,
    )
    # The count survives dataset-hash churn (the whole point)...
    assert count_variants("eurusd-logreg", ledger) == 3
    # ...and campaigns (spot vs OTC) never share a counter.
    assert count_variants("eurusd-otc-logreg", ledger) == 1
    # Every entry still records its individual dataset hash.
    entries = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert {e["dataset_content_sha256"] for e in entries} == {"hash-a", "hash-b", "hash-c"}
