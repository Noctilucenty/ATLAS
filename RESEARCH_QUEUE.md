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
8. **Frozen-universe-only policy** (evidence 2026-07-25, see below) - register
   a variant that trades ONLY the instruments the deployed model was trained
   on. 60% of live signals currently fire outside it, where no edge is
   validated and the registered verdict excludes them anyway. Pairs naturally
   with the spot-only variant (item 5's OTC finding); both are execution
   policy, not new modelling, so they are cheap to register and test.
9. **Broker ToS / automation policy review** (2026-07-24 audit item) - IQ
   Option's terms restrict automation on REAL accounts; brokers void
   winnings retroactively. MUST be answered before IQ_ALLOW_REAL is ever
   reconsidered - a policy question, not a code question. Until then the
   flag stays 0 and this is moot for the demo trial.

## Evidence gathered 2026-07-24/25 (forward window, live data)

These are MEASUREMENTS taken while the window runs, not experiments. They
sharpen the items above and none of them changed a gate.

**The meta-threshold lever has no volume at live signal rates.** Over the
first 20 forward signals the meta_p distribution was: >=0.55 kept 50%,
>=0.60 kept 25%, >=0.65 kept 5%, >=0.70 and >=0.775 kept ZERO. Median 0.551,
ceiling 0.672. The 25% keep-rate at the registered 0.60 gate matches the
decade-holdout table exactly (a good sign the meta model transfers), but the
0.70/0.775 operating points that look strongest offline produce no trades at
all here. Any future registration that leans on a high meta threshold must
budget weeks per trade, or pair it with item 4 (breadth) to buy the volume.

**EV headroom is thin.** Median EV 0.0342 against the 0.03 gate; 7/20 clear
0.04, 2/20 clear 0.05. Raising the EV gate trades volume away just as fast.

**60% of signals fire outside the trained universe.** 12 of 20 landed on
EURCHF (9), XAUUSD (2), USDCAD (1) - none in the bundle's meta['assets'] 16.
The registered candles verdict excludes them, and the model has no validated
edge there. Live orders are deliberately NOT narrowed to the 16 (that would
starve the label-fidelity measurement, which is asset-agnostic and already
volume-limited), but status/dashboard now report the split so no win rate is
ever read blended. => A spot-only, frozen-universe-only policy is a natural
candidate for the NEXT registration, alongside the OTC exclusion.

**Model confidence is structurally low, and that is the healthy signature.**
Baseline over 1,401 live cycles (pre basket-fix): median max_conf 0.0303,
mean 0.0333, stdev 0.0147. The EV gate implies a hard floor - a signal needs
p_up >= 0.5508, i.e. max_conf >= 0.0508 - so the typical cycle simply never
qualifies. Cycle-to-cycle movement averages 0.0036 (11% of level), which is
modest; the apparent jumpiness is inherent to a max taken over 39 assets.
Sustained elevation is a MARKET-CLOSED artifact: one run of 381 consecutive
cycles above the floor (~6.3h) coincides exactly with the Friday 21:00Z FX
close, when stale bars pin the model's output and the bar-freshness gate
correctly suppresses every signal anyway. Outside that run only 16 of ~1,020
open-market cycles crossed the floor (~1.6%), consistent with ~6 signals/day.
A calibrated model on 1-minute FX SHOULD sit near 0.5 (holdout Brier 0.2494
against a 0.25 coin flip); confident predictions here would indicate
overfitting, not skill. RE-MEASURE this baseline after the cross-asset basket
repair - erratic xs_mkt_vol from the 39-asset basket was feeding the model
out-of-distribution columns, so the level and variance may shift.

## Standing constraints

- One machine per account; Windows host owns the demo account now.
- `IQ_ALLOW_REAL=0` is permanent until an explicit, separate decision with
  its own risk review. Nothing in this queue touches real funds.
- Frozen pickles are never rebuilt mid-window; new models get new windows.
