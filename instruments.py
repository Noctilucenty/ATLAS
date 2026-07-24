"""Explicit broker instrument specifications.

The broker uses DIFFERENT keys for the same tradable thing depending on the
table: candles are fetched by `candle_asset`, payout and open-time are quoted
under `quote_key`, and orders are placed against `order_active`. Verified
live 2026-07-21: spot EURUSD candles/orders use 'EURUSD' while its binary
payout/openness are quoted under 'EURUSD-op'; 'EURUSD-OTC' is a SEPARATE
synthetic market (own price series, own payout) and must never be mixed with
spot - separate datasets, features, models, calibration, experiments,
reports and champions.

Openness and payout MUST both be resolved from the same `quote_key` - never
from independent fallback loops, which could silently pair one instrument's
openness with another's payout.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    candle_asset: str   # get_candles key; also the dataset asset tag
    quote_key: str      # get_all_profit / get_all_open_time key
    order_active: str   # buy() ACTIVES key
    option_kind: str    # 'turbo' or 'binary'
    expiry_minutes: int
    # Data-only instruments are collected for cross-asset context (currency
    # strength) but have no option market, so they are never traded and are
    # excluded from payout-health checks.
    tradable: bool = True


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "EURUSD": InstrumentSpec(
        candle_asset="EURUSD",
        quote_key="EURUSD-op",
        order_active="EURUSD-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURUSD-OTC": InstrumentSpec(
        candle_asset="EURUSD-OTC",
        quote_key="EURUSD-OTC",
        order_active="EURUSD-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    # Majors below verified live 2026-07-21 via iq_payouts: spot pairs quote
    # under '<PAIR>-op' (turbo + binary), OTC under '<PAIR>-OTC'. USDJPY-OTC
    # quotes binary only - no turbo market exists for it.
    "GBPUSD": InstrumentSpec(
        candle_asset="GBPUSD",
        quote_key="GBPUSD-op",
        order_active="GBPUSD-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "GBPUSD-OTC": InstrumentSpec(
        candle_asset="GBPUSD-OTC",
        quote_key="GBPUSD-OTC",
        order_active="GBPUSD-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDJPY": InstrumentSpec(
        candle_asset="USDJPY",
        quote_key="USDJPY-op",
        order_active="USDJPY-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDJPY-OTC": InstrumentSpec(
        candle_asset="USDJPY-OTC",
        quote_key="USDJPY-OTC",
        order_active="USDJPY-OTC",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "AUDUSD": InstrumentSpec(
        candle_asset="AUDUSD",
        quote_key="AUDUSD-op",
        order_active="AUDUSD-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    # AUDUSD-OTC is quoted in payouts but absent from the vendored library's
    # ACTIVES map (candles unfetchable); EURGBP-OTC is used instead.
    "EURGBP-OTC": InstrumentSpec(
        candle_asset="EURGBP-OTC",
        quote_key="EURGBP-OTC",
        order_active="EURGBP-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURJPY": InstrumentSpec(
        candle_asset="EURJPY",
        quote_key="EURJPY-op",
        order_active="EURJPY-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURJPY-OTC": InstrumentSpec(
        candle_asset="EURJPY-OTC",
        quote_key="EURJPY-OTC",
        order_active="EURJPY-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "AUDCAD-OTC": InstrumentSpec(
        candle_asset="AUDCAD-OTC",
        quote_key="AUDCAD-OTC",
        order_active="AUDCAD-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "GBPJPY-OTC": InstrumentSpec(
        candle_asset="GBPJPY-OTC",
        quote_key="GBPJPY-OTC",
        order_active="GBPJPY-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "NZDUSD-OTC": InstrumentSpec(
        candle_asset="NZDUSD-OTC",
        quote_key="NZDUSD-OTC",
        order_active="NZDUSD-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDCHF-OTC": InstrumentSpec(
        candle_asset="USDCHF-OTC",
        quote_key="USDCHF-OTC",
        order_active="USDCHF-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDSGD-OTC": InstrumentSpec(
        candle_asset="USDSGD-OTC",
        quote_key="USDSGD-OTC",
        order_active="USDSGD-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDZAR-OTC": InstrumentSpec(
        candle_asset="USDZAR-OTC",
        quote_key="USDZAR-OTC",
        order_active="USDZAR-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    # Batch 3, verified live 2026-07-22: candle fetch tested per asset and
    # quote keys read from a live get_all_profit. Spot pairs quote under
    # '<PAIR>-op'; USDCAD/EURAUD have no turbo market, only binary.
    "NZDUSD": InstrumentSpec(
        candle_asset="NZDUSD",
        quote_key="NZDUSD-op",
        order_active="NZDUSD-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDCAD": InstrumentSpec(
        candle_asset="USDCAD",
        quote_key="USDCAD-op",
        order_active="USDCAD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    # USDCHF spot: candles fetch fine but the broker quotes no option market
    # for it (no USDCHF-op key), so it is collected for CHF-strength context
    # only and never traded.
    "USDCHF": InstrumentSpec(
        candle_asset="USDCHF",
        quote_key="USDCHF-op",
        order_active="USDCHF",
        option_kind="turbo",
        expiry_minutes=1,
        tradable=False,
    ),
    "GBPJPY": InstrumentSpec(
        candle_asset="GBPJPY",
        quote_key="GBPJPY-op",
        order_active="GBPJPY-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "AUDJPY": InstrumentSpec(
        candle_asset="AUDJPY",
        quote_key="AUDJPY-op",
        order_active="AUDJPY-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURCHF": InstrumentSpec(
        candle_asset="EURCHF",
        quote_key="EURCHF-op",
        order_active="EURCHF-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "GBPCHF": InstrumentSpec(
        candle_asset="GBPCHF",
        quote_key="GBPCHF-op",
        order_active="GBPCHF-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURGBP": InstrumentSpec(
        candle_asset="EURGBP",
        quote_key="EURGBP-op",
        order_active="EURGBP-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "AUDCAD": InstrumentSpec(
        candle_asset="AUDCAD",
        quote_key="AUDCAD-op",
        order_active="AUDCAD-op",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURAUD": InstrumentSpec(
        candle_asset="EURAUD",
        quote_key="EURAUD-op",
        order_active="EURAUD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "USDHKD-OTC": InstrumentSpec(
        candle_asset="USDHKD-OTC",
        quote_key="USDHKD-OTC",
        order_active="USDHKD-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDINR-OTC": InstrumentSpec(
        candle_asset="USDINR-OTC",
        quote_key="USDINR-OTC",
        order_active="USDINR-OTC",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    # Batch 4, verified live 2026-07-24: candle fetch tested per asset. All
    # ten crosses quote BINARY-only option markets (<PAIR>-op); XAUUSD has
    # both kinds (binary payout higher). Registered post-forward-cutoff:
    # these feed the demo execution trial and reported metrics only - the
    # frozen forward-test verdict universe excludes them by construction
    # (forward_eval restricts to the model bundle's trained asset list).
    "AUDNZD": InstrumentSpec(
        candle_asset="AUDNZD",
        quote_key="AUDNZD-op",
        order_active="AUDNZD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "EURNZD": InstrumentSpec(
        candle_asset="EURNZD",
        quote_key="EURNZD-op",
        order_active="EURNZD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "GBPNZD": InstrumentSpec(
        candle_asset="GBPNZD",
        quote_key="GBPNZD-op",
        order_active="GBPNZD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "NZDJPY": InstrumentSpec(
        candle_asset="NZDJPY",
        quote_key="NZDJPY-op",
        order_active="NZDJPY-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "NZDCAD": InstrumentSpec(
        candle_asset="NZDCAD",
        quote_key="NZDCAD-op",
        order_active="NZDCAD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "CADCHF": InstrumentSpec(
        candle_asset="CADCHF",
        quote_key="CADCHF-op",
        order_active="CADCHF-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "AUDCHF": InstrumentSpec(
        candle_asset="AUDCHF",
        quote_key="AUDCHF-op",
        order_active="AUDCHF-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "CHFJPY": InstrumentSpec(
        candle_asset="CHFJPY",
        quote_key="CHFJPY-op",
        order_active="CHFJPY-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "EURCAD": InstrumentSpec(
        candle_asset="EURCAD",
        quote_key="EURCAD-op",
        order_active="EURCAD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "GBPCAD": InstrumentSpec(
        candle_asset="GBPCAD",
        quote_key="GBPCAD-op",
        order_active="GBPCAD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
    "XAUUSD": InstrumentSpec(
        candle_asset="XAUUSD",
        quote_key="XAUUSD-op",
        order_active="XAUUSD-op",
        option_kind="binary",
        expiry_minutes=1,
    ),
}


def verify_contract(
    manifest: dict, *, expiry_seconds: int, order_active: str, option_kind: str
) -> None:
    """Refuse execution when the executor's contract differs from the one the
    bundle was backtested on. A model validated on 5-minute turbo contracts
    says nothing about 1-minute or binary-kind contracts - the mismatch must
    be a hard error, never a silent substitution.

    Any model execution path MUST call this immediately before the broker
    buy call - use execution_guard.guarded_buy, never client.buy directly."""
    contract = manifest.get("contract") or {}
    if (
        contract.get("expiry_seconds") != expiry_seconds
        or contract.get("order_active") != order_active
        or contract.get("option_kind") != option_kind
    ):
        raise ValueError(
            "executor contract mismatch: bundle was validated on "
            f"{contract}, executor wants expiry_seconds={expiry_seconds}, "
            f"order_active='{order_active}', option_kind='{option_kind}'"
        )


def get_instrument(asset: str) -> InstrumentSpec:
    try:
        return INSTRUMENTS[asset]
    except KeyError:
        raise KeyError(
            f"no instrument spec for '{asset}' - add an explicit entry to "
            f"instruments.INSTRUMENTS (known: {sorted(INSTRUMENTS)})"
        ) from None

# ---------------------------------------------------------------------------
# Execution repair (2026-07-24): the broker RETIRED the legacy plain-FX
# binary actives - client.buy against them fails with "the asset is not
# available at the moment" (observed on the first 4 demo-trial orders).
# The live order books are the '-op' instruments (verified via
# get_all_init_v2: 230 binary actives, zero plain-FX names). order_active
# above therefore uses '-op' keys, and because the vendored iqoptionapi
# constants predate those actives, their server-confirmed ids are injected
# here - the one place every executor resolves order keys.
_OP_ACTIVE_IDS = {
    "AUDCAD-op": 1868, "AUDCHF-op": 1884, "AUDJPY-op": 1869,
    "AUDNZD-op": 1900, "AUDUSD-op": 1870, "CADCHF-op": 1871,
    "CHFJPY-op": 1873, "EURAUD-op": 1874, "EURCAD-op": 1875,
    "EURCHF-op": 1876, "EURGBP-op": 1862, "EURJPY-op": 1864,
    "EURNZD-op": 1901, "EURUSD-op": 1861, "GBPCAD-op": 1897,
    "GBPCHF-op": 1898, "GBPJPY-op": 1866, "GBPNZD-op": 1880,
    "GBPUSD-op": 1867, "NZDCAD-op": 1881, "NZDJPY-op": 1882,
    "NZDUSD-op": 1896, "USDCAD-op": 1878, "USDJPY-op": 1865,
    "XAUUSD-op": 1912,
}


def _inject_op_actives() -> None:
    try:
        from iqoptionapi import constants as _constants
    except Exception:  # analysis environments without the vendored library
        return
    for _key, _active_id in _OP_ACTIVE_IDS.items():
        _constants.ACTIVES.setdefault(_key, _active_id)


_inject_op_actives()
