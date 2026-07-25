"""The live cross-asset columns must match the TRAINING-time computation.

research_pooled.add_cross_asset is the reference implementation the frozen
model's features were built with; live_h2_runner.cross_asset_columns is the
one-timestamp live version. If they ever disagree, the live model is being
fed something it was not trained on - which is exactly the defect the
2026-07-24 audit found (live basket 39 vs frozen 16).
"""

import numpy as np
import pandas as pd
import pytest

from live_h2_runner import cross_asset_columns
from research_pooled import add_cross_asset

TS = 1_784_000_000
# base/quote overlap on purpose: USD, EUR, JPY and GBP all recur.
RET5 = {
    "EURUSD": 0.0010,
    "GBPUSD": -0.0004,
    "USDJPY": 0.0007,
    "EURJPY": 0.0002,
    "GBPJPY": -0.0009,
    "XAUUSD": 0.0130,   # gold: the big-vol outsider the frozen model lacks
}


def _frame(rets: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "to_ts": [TS] * len(rets),
        "asset": list(rets),
        "ret_5": [float(v) for v in rets.values()],
    })


def test_matches_training_implementation_on_the_same_basket():
    frame = _frame(RET5)
    reference = add_cross_asset(frame.copy())
    base_str, quote_str, mkt_vol = cross_asset_columns(frame, None)

    np.testing.assert_allclose(base_str, reference["xs_base_str"].to_numpy(),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(quote_str, reference["xs_quote_str"].to_numpy(),
                               rtol=0, atol=1e-12)
    assert mkt_vol == pytest.approx(float(reference["xs_mkt_vol"].iloc[0]))


def test_restricting_the_basket_matches_training_on_that_basket():
    """Scoring extra assets must not change the aggregates: the columns for
    the frozen pairs must equal what training would have produced from the
    frozen pairs alone."""
    frozen = {k: v for k, v in RET5.items() if k != "XAUUSD"}
    reference = add_cross_asset(_frame(frozen))

    full = _frame(RET5)  # gold is scored but must not contribute
    base_str, quote_str, mkt_vol = cross_asset_columns(full, set(frozen))

    for i, asset in enumerate(full["asset"]):
        if asset not in frozen:
            continue
        j = list(reference["asset"]).index(asset)
        assert base_str[i] == pytest.approx(reference["xs_base_str"].iloc[j])
        assert quote_str[i] == pytest.approx(reference["xs_quote_str"].iloc[j])
    assert mkt_vol == pytest.approx(float(reference["xs_mkt_vol"].iloc[0]))


def test_the_outsider_moves_mkt_vol_materially_when_included():
    """Guards the regression itself: including gold shifts activity enough to
    matter, which is why the basket must be pinned."""
    full = _frame(RET5)
    _, _, vol_all = cross_asset_columns(full, None)
    _, _, vol_frozen = cross_asset_columns(
        full, {k for k in RET5 if k != "XAUUSD"})
    assert vol_all > vol_frozen * 1.5


def test_non_basket_asset_is_scored_but_not_self_excluded():
    full = _frame(RET5)
    basket = {k for k in RET5 if k != "XAUUSD"}
    base_str, quote_str, _ = cross_asset_columns(full, basket)
    i = list(full["asset"]).index("XAUUSD")
    # XAU appears in no basket pair -> no strength information at all
    assert base_str[i] == 0.0
    # USD is the quote of several basket pairs, and gold contributed none of
    # them, so nothing is subtracted: it is the plain mean over contributors.
    usd_num = -RET5["EURUSD"] - RET5["GBPUSD"] + RET5["USDJPY"]
    assert quote_str[i] == pytest.approx(usd_num / 3)
