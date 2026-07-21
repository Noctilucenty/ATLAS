"""Sole gateway for any future model-driven order.

NOTHING in this repository calls guarded_buy from a live path today -
model-driven practice execution is forbidden until the promotion gates pass
prospectively (reviewer directive 2026-07-21). When an executor is
eventually approved, it MUST place orders exclusively through guarded_buy:
the run-bundle contract is re-verified immediately before the broker call,
so a mismatched expiry, active, or option kind can never reach client.buy.
"""

from instruments import InstrumentSpec, verify_contract


def guarded_buy(
    client,
    call,
    manifest: dict,
    spec: InstrumentSpec,
    amount: float,
    direction: str,
    timeout: int = 60,
):
    """Verify the bundle contract, then (and only then) place the order.

    `call` is the timeout wrapper (run_once._call) so the broker call keeps
    the same hang protection as every other library call."""
    verify_contract(
        manifest,
        expiry_seconds=spec.expiry_minutes * 60,
        order_active=spec.order_active,
        option_kind=spec.option_kind,
    )
    if direction not in ("call", "put"):
        raise ValueError(f"direction must be 'call' or 'put', got {direction!r}")
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")
    return call(
        client.buy, amount, spec.order_active, direction, spec.expiry_minutes, timeout=timeout
    )
