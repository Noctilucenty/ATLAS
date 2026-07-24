# Post-verdict research queue

Captured 2026-07-24 while the forward test runs. NOTHING here starts before
the pre-registered verdicts (forward_eval.py, once, window Jul 28 - Aug 6)
and the ~100-trade label-fidelity readout. Every item, when started, gets a
`registry.record(...)` entry FIRST (rule 2) and honest deflation. The
graveyard (FINDINGS.md §Phase 4) stays buried - nothing below re-tries it.

Goal, stated honestly: maximize *calibrated, deflated* edge - not raw WR.
Raw WR is a threshold dial (higher meta cut = higher WR, fewer trades);
chasing it without the discipline is how the 47%-OTC blind spot happened.

## Queue (rough priority)

1. **Label-fidelity consequences** - the pending ~100-trade broker-verdict vs
   candle-label measurement decides whether MID-settlement is the right
   expectation. If agreement is low, the correction feeds every future
   backtest. Blocked only by trade accumulation; analysis is ready in
   mission_control.label_fidelity.
2. **Instrument-universe expansion (spot)** - collect candles + payouts for
   IQ's remaining spot markets (metals XAUUSD/XAGUSD, indices, crypto
   binaries if quoted). Collection can start pre-verdict (data only, no
   runner change); MODELING waits. New pairs enter as a NEW registered
   hypothesis with their own forward window - never grafted onto H2.
   **STARTED 2026-07-24** (collection only): `extra_collect.py` +
   Task Scheduler `ATLAS-extra-collect` (hourly, :05) banks SpaceX-OTC,
   SpaceX-op, SP500-OTC, USSPX500, US30, USNDAQ100, UK100. First harvest:
   239 gapless candles each for the two 24/7 synthetics. Expectation
   check: SpaceX/SP500-OTC are broker-synthesized feeds - the same class
   research_otc.py measured at 47% WR for FX-OTC. Data decides later.
3. **Expiry-horizon variants** - 5m / 30m / 60m binaries as fresh registered
   hypotheses (same pipeline, new horizon + payout tables). The 15m edge
   does not transfer by assumption.
4. **Threshold economics, post-verdict** - EV-margin x meta-threshold grid
   re-registered against the NEXT forward window, sized by MinTRL from
   acceptance_report. This is the legitimate version of "tuning".
5. **Payout-aware asset ranking** - execution-level: given fixed signals,
   allocate demo orders to the highest-payout qualifying spot asset.
   Registered as an execution experiment; signal path untouched.
6. **Sizing research** - fractional-Kelly vs flat $1 on the demo account,
   using calibrated meta_p. Execution-level, registered.
7. **Second-broker portability probe** - engineering only: how much of
   collector/runner survives against another broker's API (defends against
   single-broker risk). No trading until its own registered test.

## Standing constraints

- One machine per account; Windows host owns the demo account now.
- `IQ_ALLOW_REAL=0` is permanent until an explicit, separate decision with
  its own risk review. Nothing in this queue touches real funds.
- Frozen pickles are never rebuilt mid-window; new models get new windows.
