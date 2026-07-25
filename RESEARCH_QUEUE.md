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

## EXTERNAL DEEP RESEARCH (2026-07-25, Perplexity Advanced Deep Research)

Full answer in logs/deep_research_wr.txt, raw in logs/deep_research_raw.json
(both gitignored). Cited throughout; the items below are the ones that change
what we should do. Several CONTRADICT plans we were considering.

**1. Calibration cannot rescue this, and the arithmetic says why.** Brier
skill versus a constant 0.25 forecast is 1 - 0.2494/0.25 = **0.24%**. Murphy's
decomposition (BS = reliability - resolution + uncertainty) means recalibration
lowers RELIABILITY error only; resolution rises solely from new information, a
better target, or better features. So no calibration method can manufacture
0.55-0.65 forecasts we do not already have. This kills "improve confidence by
recalibrating" as a strategy - the confidence ceiling is a resolution problem.

**2. The single highest-value change is the TARGET, not the model:** train and
validate against the broker's actual strike and expiry-settlement rule rather
than mid-price direction. Which turns out to be sharper than we assumed -

**3. SETTLEMENT IS PROBABLY NOT A SINGLE CLOSE.** Nadex (a regulated venue,
published methodology) computes FX expiration value from the last TEN
midpoints before expiry, discards the highest three and lowest three, and
averages the remaining four. If IQ does anything comparable, our label - the
close of the bar 15 bars on - is structurally wrong INDEPENDENTLY of any
markup, and a disagreement would be an averaging artefact, not evidence of
cheating. => probe_execution.py must capture several quotes around expiry, not
just the settlement bar's close, or it will misattribute the result. This is
the most concrete actionable finding in the whole report.

**4. Rejected with evidence (do not implement):** focal loss is NOT strictly
proper - its minimiser need not equal the true posterior - and on CIFAR-10 it
improved ECE 4.35%->1.48% while WORSENING error 4.95%->5.25%. Label smoothing
likewise shrinks the usable high-confidence tail. Standard split conformal is
invalid here: it assumes exchangeability, which serial dependence and regime
shifts violate, and treating singleton prediction sets as trades does NOT
guarantee the singleton subset achieves the nominal level.

**5. Venn-Abers is the one calibration method worth testing** (largest average
log-loss reduction in a 2026 tabular benchmark, beta calibration close behind)
- but explicitly "cannot create tail observations or improve their rank". It
makes the EV gate HONEST; it does not add trades. Use cross-Venn-Abers pooled
across instruments, never another small per-pair isotonic fit.

**6. Feature families worth testing post-verdict, all computable from 1m OHLC**
and none on the graveyard list: close-location value and wick imbalance;
path efficiency ER_k = |sum r| / sum|r| with sign-change counts and run
lengths; signed semivariance imbalance (RS+ - RS-)/(RS+ + RS-); range-surprise
x close-location interaction; cross-pair currency residuals. Honest caveat
from the report: no published work shows a stable 57-65% out-of-sample win
rate for 15-minute FX binaries from 1m OHLC after settlement effects - the
nearest credible intraday-FX evidence sits at 51-60%.

**7. Our power arithmetic may be optimistic.** A Bernoulli SPRT against
p0 = 0.53476 needs roughly 760 independent trades to accept p = 0.57, and ~220
at p = 0.60 - versus MinTRL 163. At ~6 signals/day that is ~127 active trading
days. Overlapping expiries extend it further.

**8. A better instrument than our one pre-committed extension exists:**
confidence sequences / e-processes (beta-binomial mixture likelihood ratio
against p <= 0.53476, updated per independent expiry cluster) are ALWAYS-VALID
under continuous monitoring - P(p_tau <= alpha) <= alpha at every stopping
time. That permits legitimate continuous monitoring rather than a single
pre-committed extension date. Cannot replace the registered criteria for THIS
window, but is the right design for the next registration.

**9. Selecting the best of 39 instruments each minute IS a selection effect.**
The report warns the maximum of 39 noisy probabilities must not be mistaken
for confidence, and that the whole policy including asset choice should be
evaluated per timestamp. Directly relevant: our max_conf dashboard metric is
exactly that maximum.

## Standing constraints

- One machine per account; Windows host owns the demo account now.
- `IQ_ALLOW_REAL=0` is permanent until an explicit, separate decision with
  its own risk review. Nothing in this queue touches real funds.
- Frozen pickles are never rebuilt mid-window; new models get new windows.
