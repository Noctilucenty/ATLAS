"""Preflight must be outcome-blind.

Its whole justification is that it can run before verdict day WITHOUT
consuming the single permitted evaluation. That holds only if it never reads a
label - so these tests pin the clustering math and the absence of any outcome
read in the preflight code path.
"""

from pathlib import Path

from forward_eval import count_clusters

PROJECT_DIR = Path(__file__).resolve().parent.parent
PURGE = 15 * 60


def test_count_clusters_chains_overlapping_trades():
    # three trades inside one purge window -> one cluster
    ts = [1000, 1000 + 300, 1000 + 600]
    assert count_clusters(ts, PURGE) == (3, 1)


def test_count_clusters_separates_distant_trades():
    ts = [1000, 1000 + PURGE + 1, 1000 + 2 * (PURGE + 1)]
    assert count_clusters(ts, PURGE) == (3, 3)


def test_count_clusters_is_order_independent():
    ts = [5000, 1000, 3000]
    assert count_clusters(ts, PURGE) == count_clusters(sorted(ts), PURGE)


def test_count_clusters_empty():
    assert count_clusters([], PURGE) == (0, 0)


def test_count_clusters_matches_cluster_stats_on_the_same_timestamps():
    """The count-only clusterer must agree with the real one, or preflight
    would misreport the extension trigger."""
    from forward_eval import cluster_stats

    ts = [100, 400, 1300, 1400, 9000]
    trades = [("EURUSD", t, True, 0.87) for t in ts]
    real = cluster_stats(trades, PURGE)
    ind, clus = count_clusters(ts, PURGE)
    assert clus == real["clusters"]


def test_preflight_never_reads_labels():
    """label_up must not appear anywhere in preflight's body - reading it
    would turn a readiness check into a peek at the result."""
    src = (PROJECT_DIR / "forward_eval.py").read_text(encoding="utf-8")
    body = src.split("def preflight(")[1].split("\ndef _frozen_assets")[0]
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))
    # the docstring mentions label_up while explaining the guarantee; strip it
    code = code.split('"""')[-1]
    assert "label_up" not in code, (
        "preflight reads label_up - it would no longer be outcome-blind")
    for banned in ("won", "cluster_win_frac", "p_one_sided", "verdict("):
        assert banned not in code, f"preflight computes an outcome: {banned}"
