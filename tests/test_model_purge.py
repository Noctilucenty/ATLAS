"""The per-model purge: no bundle may ever be scored on its own training
labels.

Labels look `horizon` bars forward, so a model whose data ends at T has
absorbed prices through T + horizon. The purge is derived from the artifacts
(data_end_ts + horizon), so it carries no outcome-informed choice - which is
what makes it legitimate to apply mid-window.
"""

import pytest

from forward_eval import model_forward_start

CUTOFF = 1784678400          # 2026-07-22T00:00:00Z
HORIZON = 15


def test_clean_model_starts_at_the_registration_cutoff():
    """A bundle trained entirely before the cutoff imposes no extra purge."""
    meta = {"data_end_ts": CUTOFF - 3600}
    assert model_forward_start(meta, CUTOFF, HORIZON) == CUTOFF


def test_contaminated_model_starts_one_horizon_after_its_data_end():
    # h2: data_end 2026-07-22T05:04Z -> scoring starts 05:19Z
    h2_end = 1784696640
    start = model_forward_start({"data_end_ts": h2_end}, CUTOFF, HORIZON)
    assert start == h2_end + HORIZON * 60
    assert start == 1784697540          # 2026-07-22T05:19:00Z


def test_h4_gets_a_later_start_than_h2():
    """H4 trained 12.5h longer, so it must be purged further - scoring it
    from H2's start would score it on its own training labels."""
    h2 = model_forward_start({"data_end_ts": 1784696640}, CUTOFF, HORIZON)
    h4 = model_forward_start({"data_end_ts": 1784741520}, CUTOFF, HORIZON)
    assert h4 > h2
    assert h4 == 1784741520 + HORIZON * 60   # 2026-07-22T17:47:00Z


def test_one_horizon_is_actually_added_not_just_the_data_end():
    """The bug this guards: using data_end_ts itself would still score the
    model on the label it learned from at that bar."""
    end = CUTOFF + 600
    start = model_forward_start({"data_end_ts": end}, CUTOFF, HORIZON)
    assert start != end, "purge must exceed data_end_ts by the label horizon"
    assert start - end == HORIZON * 60


def test_missing_data_end_falls_back_to_the_cutoff():
    """meta-h3.pkl has no data_end_ts; with nothing to derive from, the
    registration cutoff stands (and load_bundle warns separately)."""
    assert model_forward_start({}, CUTOFF, HORIZON) == CUTOFF


def test_horizon_scales_the_purge():
    end = CUTOFF + 60
    assert (model_forward_start({"data_end_ts": end}, CUTOFF, 5)
            == end + 5 * 60)
    assert (model_forward_start({"data_end_ts": end}, CUTOFF, 30)
            == end + 30 * 60)


def test_purge_is_applied_in_candles_track_source():
    """Structural guard: load_bundle no longer refuses a post-cutoff bundle
    when purge=True, so the purge MUST be applied or contaminated rows would
    be scored silently."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "forward_eval.py").read_text(
        encoding="utf-8")
    track = src.split("def candles_track")[1].split("def _frozen_assets")[0]
    assert "h2_start = model_forward_start(" in track
    assert 'pooled["to_ts"] > h2_start' in track, (
        "forward set is not filtered by the purged start")
    assert "h4_start = model_forward_start(" in track
    assert "ts <= h4_start" in track, "H4 rows are not purged by its own start"
