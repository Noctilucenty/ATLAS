import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import FEATURE_COLUMNS, FEATURE_VERSION, build_features  # noqa: E402

START = 1_784_500_000 - (1_784_500_000 % 60)


def make_frame(closes: list[float], interval: int = 60) -> pd.DataFrame:
    rows = []
    prev = closes[0]
    for i, c in enumerate(closes):
        high = max(prev, c) * 1.0001
        low = min(prev, c) * 0.9999
        rows.append(
            {
                "from_ts": START + i * interval,
                "to_ts": START + (i + 1) * interval,
                "open": prev,
                "high": high,
                "low": low,
                "close": c,
                "volume": 100.0 + (i % 7),
            }
        )
        prev = c
    return pd.DataFrame(rows)


def wave(n: int = 900) -> list[float]:
    return [1.10 + 0.002 * np.sin(i / 25) + 0.0004 * np.sin(i / 7) for i in range(n)]


def test_output_has_all_feature_columns_and_version():
    out = build_features(make_frame(wave()), horizon=5)
    for col in FEATURE_COLUMNS + ["from_ts", "to_ts", "label_up", "feature_version"]:
        assert col in out.columns, col
    assert (out["feature_version"] == FEATURE_VERSION).all()
    # Warmup is dominated by the 15m EMA(21) trend: ~21*15 = 315 base bars.
    assert len(out) > 500
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_deterministic():
    a = build_features(make_frame(wave()), horizon=5)
    b = build_features(make_frame(wave()), horizon=5)
    pd.testing.assert_frame_equal(a, b)


def test_no_lookahead_in_features():
    """Features computed on a truncated series must equal the same rows of the
    full series - i.e. row t never uses rows after t."""
    full_frame = make_frame(wave())
    cut = 700
    full = build_features(full_frame, horizon=5)
    trunc = build_features(full_frame.iloc[:cut], horizon=5)
    merged = full.merge(trunc, on="from_ts", suffixes=("_full", "_trunc"))
    assert len(merged) > 100
    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            merged[f"{col}_full"].to_numpy(),
            merged[f"{col}_trunc"].to_numpy(),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"lookahead detected in feature {col}",
        )


def test_labels_look_forward_correctly():
    closes = [1.0] * 100 + [1.0 + 0.001 * i for i in range(1, 51)]  # flat then rising
    out = build_features(make_frame(closes), horizon=5)
    rising = out[out["from_ts"] >= START + 100 * 60]
    assert (rising["label_up"].dropna() == 1.0).all()


def test_tie_label_is_nan_and_end_rows_unlabeled():
    closes = wave(800)
    frame = make_frame(closes)
    out = build_features(frame, horizon=5)
    # Last horizon rows have no future close -> NaN label.
    assert out["label_up"].tail(5).isna().all()
    # A perfectly flat stretch produces tie labels -> NaN.
    flat = build_features(make_frame([1.1] * 700), horizon=5)
    assert flat["label_up"].isna().all()


def test_sessions_are_mutually_exclusive_within_covered_hours():
    out = build_features(make_frame(wave()), horizon=5)
    total = out["session_asia"] + out["session_europe"] + out["session_us"]
    assert set(total.unique()) <= {0.0, 1.0}


def test_mtf_align_only_takes_trend_values():
    out = build_features(make_frame(wave()), horizon=5)
    assert set(out["mtf_align"].unique()) <= {-1.0, 0.0, 1.0}


def test_real_dataset_if_available():
    """Smoke-run the pipeline over the live-collected canonical history
    (same loader production train.py uses)."""
    db_file = Path(__file__).resolve().parent.parent / "market.duckdb"
    if not db_file.exists():
        pytest.skip("no collected dataset")
    from storage import load_canonical_history, open_db

    try:
        conn = open_db(db_file)
    except Exception as exc:
        # A collector or backfill run holds a write lock; that is normal
        # operation, not a test failure for an "if available" smoke test.
        pytest.skip(f"dataset busy: {type(exc).__name__}")
    candles, report = load_canonical_history(conn, "EURUSD", 60)
    if report["candles"] < 600:
        pytest.skip("not enough accumulated history yet")
    out = build_features(candles, horizon=5)
    assert len(out) > 300
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_zero_volume_feed_yields_neutral_vol_rel():
    """IQ Option OTC candles report volume=0 on every bar; the frame must
    still produce rows, with vol_rel pinned to the neutral 1.0."""
    frame = make_frame(wave())
    frame["volume"] = 0.0
    out = build_features(frame, horizon=5)
    assert len(out) > 500
    assert (out["vol_rel"] == 1.0).all()
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_tiny_segments_are_skipped_not_crashed():
    """Contiguous segments shorter than indicator warmup (common in external
    deep-history data) must contribute nothing rather than crash ta's ATR."""
    frame = make_frame(wave(30))
    frame.loc[4:, "from_ts"] += 3600  # gap after 4 bars -> 4-bar segment
    frame.loc[4:, "to_ts"] += 3600
    out = build_features(frame, horizon=5)
    assert len(out) == 0


def test_extra_vol_features_are_optional_and_causal():
    """extra_vol=True adds range-based estimators without disturbing the
    default contract, and row t must never use rows after t."""
    from features import EXTRA_VOL_COLUMNS

    frame = make_frame(wave())
    base = build_features(frame, horizon=5)
    assert not any(c in base.columns for c in EXTRA_VOL_COLUMNS)

    full = build_features(frame, horizon=5, extra_vol=True)
    for col in EXTRA_VOL_COLUMNS:
        assert col in full.columns, col
    assert not full[EXTRA_VOL_COLUMNS].isna().any().any()
    assert (full["cs_spread"] >= 0).all()
    assert (full[["gk_vol", "rs_vol", "park_vol"]] >= 0).all().all()

    # Truncating the input must not change the surviving rows' values.
    cut = 700
    truncated = build_features(frame.iloc[:cut].copy(), horizon=5, extra_vol=True)
    shared = full[full["to_ts"].isin(truncated["to_ts"])]
    merged = truncated.merge(shared, on="to_ts", suffixes=("_t", "_f"))
    for col in EXTRA_VOL_COLUMNS:
        pd.testing.assert_series_equal(
            merged[f"{col}_t"], merged[f"{col}_f"], check_names=False
        )
