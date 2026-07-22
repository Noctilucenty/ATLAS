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
        order_active="EURUSD",
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
        order_active="GBPUSD",
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
        order_active="USDJPY",
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
        order_active="AUDUSD",
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
        order_active="EURJPY",
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
        order_active="NZDUSD",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "USDCAD": InstrumentSpec(
        candle_asset="USDCAD",
        quote_key="USDCAD-op",
        order_active="USDCAD",
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
        order_active="GBPJPY",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "AUDJPY": InstrumentSpec(
        candle_asset="AUDJPY",
        quote_key="AUDJPY-op",
        order_active="AUDJPY",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURCHF": InstrumentSpec(
        candle_asset="EURCHF",
        quote_key="EURCHF-op",
        order_active="EURCHF",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "GBPCHF": InstrumentSpec(
        candle_asset="GBPCHF",
        quote_key="GBPCHF-op",
        order_active="GBPCHF",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURGBP": InstrumentSpec(
        candle_asset="EURGBP",
        quote_key="EURGBP-op",
        order_active="EURGBP",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "AUDCAD": InstrumentSpec(
        candle_asset="AUDCAD",
        quote_key="AUDCAD-op",
        order_active="AUDCAD",
        option_kind="turbo",
        expiry_minutes=1,
    ),
    "EURAUD": InstrumentSpec(
        candle_asset="EURAUD",
        quote_key="EURAUD-op",
        order_active="EURAUD",
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
