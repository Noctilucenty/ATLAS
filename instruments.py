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
}


def get_instrument(asset: str) -> InstrumentSpec:
    try:
        return INSTRUMENTS[asset]
    except KeyError:
        raise KeyError(
            f"no instrument spec for '{asset}' - add an explicit entry to "
            f"instruments.INSTRUMENTS (known: {sorted(INSTRUMENTS)})"
        ) from None
