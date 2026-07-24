"""Execution-repair guards (2026-07-24): tradable spot orders must resolve
to live '-op' actives, and their ids must be present in the ACTIVES map the
buy() path uses. The legacy plain-FX actives are retired broker-side -
ordering against them fails with 'the asset is not available at the moment'."""

import pytest

from instruments import _OP_ACTIVE_IDS, INSTRUMENTS

iq_constants = pytest.importorskip("iqoptionapi.constants")


def _tradable_spot():
    return {a: s for a, s in INSTRUMENTS.items()
            if s.tradable and not a.endswith("-OTC")}


def test_all_tradable_spot_orders_use_op_actives():
    specs = _tradable_spot()
    assert specs, "no tradable spot instruments?"
    for asset, spec in specs.items():
        assert spec.order_active.endswith("-op"), (
            f"{asset}: order_active={spec.order_active!r} targets a retired "
            "legacy active - must be the '-op' instrument")


def test_op_active_ids_are_resolvable_for_buy():
    for asset, spec in _tradable_spot().items():
        assert spec.order_active in iq_constants.ACTIVES, (
            f"{asset}: {spec.order_active!r} missing from ACTIVES - buy() "
            "would KeyError")
        assert iq_constants.ACTIVES[spec.order_active] == \
            _OP_ACTIVE_IDS[spec.order_active]


def test_otc_actives_unchanged():
    # OTC ids were verified live-correct; the repair must not touch them.
    for asset, spec in INSTRUMENTS.items():
        if asset.endswith("-OTC"):
            assert spec.order_active == asset
