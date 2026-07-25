"""The evaluator must implement the REGISTRATION, not its own variant.

Each value pinned here was fixed by FORWARD_TEST.md before any forward data
existed, so these tests encode the registration - not a preference. They exist
because three mismatches were found days before the single permitted run.
"""

from pathlib import Path

import pytest

import forward_eval as fe

PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_min_clusters_matches_the_registered_twenty():
    """H2 primary, H2 secondary, H3 and H4 are all registered at ">= 20
    clusters". The >= 30 in the Protocol section belongs to the older H1."""
    assert fe.MIN_CLUSTERS_CANDLES == 20
    assert fe.MIN_CLUSTERS_PAPER == 20


def test_alpha_is_bonferroni_over_the_four_registered_tests():
    assert fe.ALPHA == pytest.approx(0.05 / 4)
    assert fe.REGISTERED_VERDICT_KEYS == {
        "h2_primary", "h2_secondary", "h3_meta60", "h4"}


def test_registered_gates_and_threshold():
    assert fe.H2_PRIMARY_MARGIN == 0.03
    assert fe.H2_SECONDARY_MARGIN == 0.04
    assert fe.H3_META_THRESHOLD == 0.60      # ORIGINAL registration, not 0.65


def test_observed_payout_prefers_binary_like_the_runner():
    """The contract traded is a 15-minute BINARY and the frozen config says
    "binary-kind payout", but spec.option_kind is 'turbo' for most
    instruments. The evaluator must price the contract that was traded."""
    source = (PROJECT_DIR / "forward_eval.py").read_text(encoding="utf-8")
    fn = source.split("def observed_payout")[1].split("\n    for label")[0]
    # Compare the executable BODY only - the docstring mentions option_kind
    # while explaining the fix, which would confound a textual ordering check.
    body = fn.split('"""')[-1]
    calls = [ln for ln in body.splitlines() if "latest_payout_before" in ln]
    assert calls, "observed_payout makes no payout lookup"
    joined = "\n".join(body.splitlines())
    assert '"binary"' in joined, "observed_payout never queries the binary payout"
    # binary must be asked FIRST; a turbo fallback afterwards is fine
    assert joined.index('"binary"') < (
        joined.index("spec.option_kind")
        if "spec.option_kind" in joined else len(joined))


def test_paper_track_filters_to_the_frozen_universe():
    source = (PROJECT_DIR / "forward_eval.py").read_text(encoding="utf-8")
    paper = source.split("def paper_track")[1]
    assert "_frozen_assets()" in paper, (
        "paper_track issues registered verdicts without the frozen-universe "
        "filter that candles_track applies")


def test_verdict_requires_both_beating_breakeven_and_significance():
    # 'independent' at/above MinTRL so the power rule does not intervene
    powered = {"clusters": 25, "cluster_win_frac": 0.60, "breakeven": 0.5348,
               "p_one_sided": 0.001, "independent": 200}
    assert fe.verdict(powered, 20) == "PASS"
    # significant but below break-even -> FAIL (the window could detect it)
    assert fe.verdict({**powered, "cluster_win_frac": 0.50}, 20) == "FAIL"
    # beats break-even but not significant at the Bonferroni alpha -> FAIL
    assert fe.verdict({**powered, "p_one_sided": 0.02}, 20) == "FAIL"
    # too few clusters -> INCONCLUSIVE, never FAIL
    assert "INCONCLUSIVE" in fe.verdict({**powered, "clusters": 19}, 20)
    assert "INCONCLUSIVE" in fe.verdict({"clusters": 0}, 20)


def test_a_fail_below_mintrl_is_reported_as_underpowered_not_refutation():
    """PRE-COMMITTED POWER RULE 1: below MinTRL a losing result cannot
    distinguish 'no edge' from 'not enough data', so it must not read as
    refutation. A PASS is unaffected - it still requires what was registered."""
    underpowered = {"clusters": 22, "cluster_win_frac": 0.50,
                    "breakeven": 0.5348, "p_one_sided": 0.40,
                    "independent": 30}
    out = fe.verdict(underpowered, 20)
    assert "UNDERPOWERED" in out and "not refutation" in out
    assert out != "FAIL"

    # the same shortfall must NOT downgrade a genuine PASS
    passing = {**underpowered, "cluster_win_frac": 0.70, "p_one_sided": 0.001}
    assert fe.verdict(passing, 20) == "PASS"


def test_mintrl_anchor_matches_the_registered_table():
    assert fe.MINTRL_AT_62 == 163
