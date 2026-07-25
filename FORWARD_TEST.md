# Pre-registered forward test — pooled cross-asset model

Registered: 2026-07-22 (UTC). Do not modify the hypothesis or config after
this date; a failed forward test may not be re-run with tweaked parameters
against the same forward window.

## In-sample evidence (why this test exists)

Pooled walk-forward (`research_pooled.py --horizon 15`, features v1.3.0,
668,342 rows, 10 instruments, time-purged folds, data 2026-05-21..07-21):

- 462 raw EV-gated trades → 130 per-asset independent → 83 wins (63.9%, p=0.002)
- Cross-asset chain clustering (the stricter test): 70 clusters,
  mean cluster win fraction 0.675 (t-test p=0.0007), majorities 47W/22L
  (sign test p=0.0035), spread over 17 distinct dates
- All 5 fold Briers below 0.250 (0.24979–0.25001)
- Caveat: ~20 experiment variants were explored this session before this
  result; treat all in-sample p-values as optimistic. Hence this document.

## Frozen configuration

- Model: LightGBM, `LGBM_DEFAULTS` in train.py (NOT tuned), wrapped in
  ChronoCalibratedModel (n_folds=3, gap=150 rows pooled)
- Features: FEATURE_VERSION 1.3.0, all FEATURE_COLUMNS, asset-agnostic
- Pool: all 10 registered instruments, 60s candles
- Label horizon: 15 bars (15 minutes)
- Decision: EV policy, ev_margin=0.02, binary-kind payout (assume 0.87 only
  if no causal snapshot; prefer prospective payouts once snapshots cover the
  window)
- Train window: all data up to the forward-test start (2026-07-22)

## Protocol

1. Accumulate ≥14 calendar days of NEW candles + hourly payout snapshots
   (collector must be running; forward data must post-date this file).
2. Train once on all data before 2026-07-22; predict the forward window once.
3. Evaluate exactly as in-sample: per-asset independent trades AND
   cross-asset clusters. Success criterion, chosen in advance: cluster mean
   win fraction > break-even at the OBSERVED prospective payouts, one-sided
   t-test p < 0.05, with ≥ 30 clusters. Anything less is a fail.
4. Report the result either way. A fail closes this hypothesis; do not
   iterate against the forward window.

## Hypothesis #2 (registered 2026-07-22, before any forward data existed)

Same protocol and forward window as #1. Config: pooled LightGBM over ALL 16
registered instruments, features v1.3.0 + cross-asset currency-strength
columns (research_pooled.py --cross-asset), REALISTIC labels
(--entry-next-open: strike = next bar's open), horizon 15.

In-sample (1,105,496 rows, data ..2026-07-21), by EV gate:

| ev_margin | independent trades | win rate | clusters | cluster win frac |
|---|---|---|---|---|
| 0.02 | 105 | 66.7% | 56 | 69.2% (p=0.0006) |
| 0.03 | 64 | 75.0% | 39 | 76.4% (p=0.0002) |
| 0.04 | 41 | 80.5% | 22 | 81.1% (p=0.0006) |
| 0.05 | 20 | 90.0% | 13 | 98.3% |

The monotone gate->win-rate curve is what a genuinely calibrated edge looks
like; it survived realistic entry pricing and 70% new data vs hypothesis #1.

Pre-committed forward evaluation for #2: PRIMARY gate ev_margin=0.03
(secondary 0.02 reported alongside); success = cluster mean win fraction
above observed-payout break-even, one-sided t-test p < 0.05, >= 20 clusters.
The gate choice is fixed NOW - picking the best-looking margin after seeing
forward results is forbidden.

Secondary POLICY variant (registered 2026-07-22, same forward window):
gate ev_margin=0.04 - the decade anchors show the staircase is monotone, so
0.04 should trade ~30% less at ~1-2 points higher win rate. Reported
alongside the primary, never substituted for it after the fact.

## Hypothesis #3 (registered 2026-07-22, before any forward data existed)

H2's signals filtered by a META-LABELING model (research_meta.py): an LGBM
trained on 42,619 decade gated trades (2016-2022, EURUSD/GBPUSD/USDJPY
histdata) to predict whether a gated trade wins, from trade context (hour,
volatility, trend strength, model confidence). Honest holdout (2023-2025,
13,237 trades, base 55.9%):

| meta threshold | kept | win rate | lift |
|---|---|---|---|
| 0.55 | 66% | 57.4% | +1.5 (p=0.005) |
| 0.60 | 25% | 61.9% | +6.0 (p~0) |
| 0.65 | 10% | 68.2% | +12.4 (p~0) |

Independently replicated gating fact: late-UTC session (21-24h) wins 60.7%
on holdout vs ~54% for all other sessions. ADX tables did NOT replicate
between periods and are not used.

Pre-committed forward evaluation for #3: H2 primary-gate signals with
meta_p >= 0.60 (deployed model models/meta-h3.pkl, refit on all 55,856
decade trades; threshold fixed NOW). Success criterion as #2. Caveat noted
in advance: the meta model is spot-trained; its transfer to OTC instruments
is untested and the forward test will answer it.

## Deep-history anchor (2026-07-22, research_deephistory.py)

Same feature family (v1.3.0), model, horizon and REALISTIC labels on ten
years of histdata.com spot EURUSD (2016-2025, 2,475,914 labeled rows,
8 time-purged folds):

- Every fold's Brier below 0.250 (mean 0.24939)
- ev 0.02: 19,016 independent trades, 56.9% win rate
- ev 0.03: 13,215 independent trades, 57.6%
- ev 0.04:  9,244 independent trades, 58.6% (monotone in the gate again)
- Break-even at 0.87 payout: 53.5%

Replicated on ten years each of GBPUSD (22,202 independent trades at
ev 0.02, 56.4% -> 58.9% monotone in the gate, mean Brier 0.24935) and
USDJPY (14,638 trades, 55.7% -> 58.1%, mean Brier 0.24940). Three majors,
~7.4M labeled rows, identical signature.

Interpretation: the edge signature is stable across a decade and every
market regime in it - this is not a 2026-specific artifact. Remaining gaps
to live profitability are execution-side: histdata carries no spread, and
IQ Option's book/fills may differ. The forward test remains the referee.

## Range-volatility ablation (2026-07-22, decade histdata, --extra-vol)

Adding gk_vol/rs_vol/park_vol/cs_spread/vol_of_vol to v1.3.0, independent
gated-trade win rate, baseline -> with extra_vol:

| pair | ev 0.02 | ev 0.03 | ev 0.04 |
|---|---|---|---|
| EURUSD | 56.9 -> 57.4 | 57.6 -> 58.2 | 58.6 -> 59.3 |
| GBPUSD | 56.4 -> 57.0 | 57.3 -> 57.9 | 58.9 -> 59.9 |
| USDJPY | 55.7 -> 56.6 | 56.6 -> 58.1 | 58.1 -> 58.7 |

9/9 comparisons improve (+0.5 to +1.5, mean ~+0.7) across three
independent pairs and a decade each - a real effect, not a single-pair
fluke. Trade count falls ~15% (the features make the model more selective).

Counter-evidence, recorded deliberately: cross-asset CLUSTER win fraction
falls on all three pairs (-0.4, -1.4, -0.3). The two metrics estimate
different things - independent-trade win rate models "trade one at a time
per asset", which is how we would actually trade, while the cluster metric
weights each burst equally and exists as a significance guard. The
divergence means extra_vol wins more of the trades we would take but
reshapes burst structure slightly unfavourably. Adopted as a candidate,
NOT retrofitted into the frozen H2/H3 models.

## Best-option search (2026-07-22, research_best.py, leak-free)

The meta-threshold dial extends monotonically past 0.70: on the pooled
global bundle the holdout staircase is 73.0 / 75.8 / 77.2 / 80.9% at
0.70 / 0.725 / 0.75 / 0.775 (392 independent trades at the top; 0.80 goes
thin). The 0.775 operating point passes the full acceptance contract
(deflated z 8.2 after a 133-trial penalty, PBO 0.00) and its MinTRL is ~13
trades. It is reported on the forward window as an EXPLORATORY metric only;
the primary hypothesis remains H3 @ 0.65 and alpha accounting is unchanged.

Breadth scaling (9-pair pooled decade, registered 2026-07-23): pooling 9
majors instead of 3 leaves the win-rate staircase unchanged (80.6 vs 80.9%
at meta 0.775) but multiplies independent trades ~2.7x at every threshold
(1,040 vs 392 at the top). Breadth scales volume, not the ceiling - which
validates the deployed 16-instrument pooled architecture and shortens any
future validation window.

Tested against it and REJECTED (all leak-free, same bundles):
- meta v2 (enriched context: range-vol + H1/H4 positional + causal
  per-asset streak features + enriched-model probability): 78.2% vs v1's
  79.1% at 0.775 - no gain.
- consensus gate (base and enriched direction models must agree): produced
  byte-identical trade sets at every threshold - surviving high-meta trades
  already agree; zero effect.
- per-pair isotonic recalibration: rescales the threshold axis without
  improving selection at matched volume; useful for stake sizing someday,
  not for win rate.

## Era holdout 2003-2015 (pre-registered 2026-07-23, run once)

The only data our 138 experiments never touched: thirteen years downloaded
AFTER the expectation was registered. Frozen recipe, era-internal
walk-forward, meta trained <= 2010, scored once on 2011-2015 at thresholds
fixed from the modern work.

Verdict per the pre-committed criteria: REPRODUCED - monotone and above
break-even at all three thresholds (56.7% / 57.1% / 57.1% on 14,468 /
9,375 / 3,204 independent trades; worst p = 2e-5).

THE HONEST READING, recorded prominently: the magnitude is far smaller
than on 2016-2025 (67-79%). The staircase rises 0.4pt across thresholds in
that era versus 12pt on modern data - the edge EXISTS everywhere tested,
but the strong modern win rates are partly era-specific and/or inflated by
research-process contamination of 2016-2025 despite the selection/holdout
discipline. The era result is the LEAST-contaminated estimate we own.
Forward expectations should be anchored nearer 57-65% than 79%. At 0.87
payout, 57% is still +6.6% EV per trade - profitable, not spectacular.

## Rejected levers (2026-07-22) - recorded so they are not re-tried

DAY-OF-WEEK (2026-07-24): dow_sin/cos added to the direction model, decade
EURUSD like-for-like: -0.4/+1.6/-0.7/+0.6pt across meta thresholds -
alternating sign, pure noise. Hour-of-day plus the meta context already
absorb calendar structure. REJECTED.

Both tested on decade EURUSD, identical machinery and row counts.

ENSEMBLE (3-seed LGBM + logreg averaging): independent win rate 56.6 / 57.3
/ 59.0 vs baseline 56.9 / 57.6 / 58.6 - no win-rate gain. Cluster win
fraction 70.1 vs 67.5 and 25% more trades at equal accuracy. NOT adopted as
a win-rate lever; its real value is statistical power, which is worth
revisiting only if the forward window returns INCONCLUSIVE for lack of
trades.

HAR-RV realized-volatility term structure: 57.1 / 57.2 / 58.8 vs baseline
56.9 / 57.6 / 58.6 - a wash on win rate, worse cluster fraction (65.2 vs
67.5), worse Brier, and 43% fewer trades. REJECTED. Volatility forecasting
predicts magnitude, not direction, and the data agrees with the theory.

Methodology note: the FIRST HAR run appeared to be a decisive loss (55.3% at
the primary gate) and matched the prediction made before running it. It was
invalid - a 1-day RV window needs 1440 contiguous bars and this data
fragments into 12,117 segments with median length 24, so 85% of rows were
lost to warmup and the model trained on 530k rows against 2.48M. A result
that confirms a prior is exactly when the mechanism deserves checking.

## Hypothesis #4 (registered 2026-07-22, before any forward data existed)

H2's configuration plus the extra_vol feature block, same primary gate
(ev_margin 0.03), same success criterion. Requires its own frozen model
(live_model_build.py --extra-vol) before the forward window is scored.

## MULTIPLICITY CORRECTION (registered 2026-07-22)

Four pre-registered tests now share ONE forward window (H2 primary, H2
secondary, H3, H4). Testing four hypotheses at p < 0.05 gives roughly a
19% chance that at least one passes by luck alone. To keep the family-wise
error at 5%, each test must clear the Bonferroni threshold

    p < 0.05 / 4 = 0.0125

H3 (meta-filtered) is designated the SINGLE PRIMARY hypothesis; the other
three are secondary and reported for information. A secondary passing
while H3 fails is NOT a green light to trade - it is a new hypothesis
requiring its own fresh forward window.

## Acceptance framework (added 2026-07-22; does not alter any frozen hypothesis)

An offline acceptance contract (acceptance_report.py) now gates any future
candidate: leak-free holdout edge at the Bonferroni alpha, PBO < 0.40
(CSCV), deflated win rate penalised for the HONEST experiment count
(research_registry.jsonl - 98 variants attempted as of registration),
Brier < 0.25, meta ECE < 0.05, >= 200 independent holdout trades, and a
paper check that ONLY the pre-registered forward test can set.

Global-model candidate (ev 0.03, meta 0.65) status: PROVISIONAL PASS -
69.5% on 1,981 independent holdout trades, PBO 0.00 over 70 combinations,
deflated z 11.8 after the 98-trial penalty, meta ECE 0.023 (the meta model
runs ~2-3pt overconfident on USDJPY - noted, within gate). Paper: PENDING.

MinTRL at the Bonferroni alpha sizes the forward window: 163 independent
trades suffice if the true win rate is 62%, 72 if 66%, 39 if 70%.

Shadow tracks (Farxida-style, same timestamps/prices/payouts, all frozen):
the live log's p_up/meta_p fields already encode the no-meta track (H2
primary), every meta-threshold track, and the INVERTED track (flip each
action at scoring time); the runner additionally logs h4_p (extra-vol
shadow) and decision_latency_s per signal. The no-model baseline is
break-even by construction. None of these gates or trades anything.

## Execution-measurement track (started 2026-07-24, $1 PRACTICE trades)

The live paper agent now runs with --trade: every H2-primary-gate signal
places a $1 PRACTICE 15-minute binary (hard PRACTICE-only guard;
IQ_ALLOW_REAL=0). Started BEFORE the forward verdict on purpose: demo
trades cannot contaminate the hypothesis evaluation (signals and their
candle-scored outcomes are unchanged) and execution friction is a
multiplicative unknown on top of the statistical edge - the standard way
backtested edges die in the wild. Measuring it concurrently shortens the
road to a real conclusion.

What this track measures, per trade: broker outcome (check_win_v4) vs our
candle-label outcome (agreement rate = label fidelity), realised payout vs
quoted, order rejections, and expiry-alignment effects (broker binary
expiries snap to clock boundaries; our labels assume 15 minutes from the
signal bar - the disagreement rate IS the measurement). Stakes are $1 on a
resettable demo balance; ~6 trades/day expected at the 0.03 gate.

This track is a MEASUREMENT, not a hypothesis: it has no pass/fail
criteria and cannot influence the pre-registered verdicts. Its results
gate only the LATER decision of whether candle-based win rates translate
to executable win rates.

## OTC finding (2026-07-24, research_otc.py, pre-cutoff broker data)

The decade validations were spot-only while 12+ of the 28 live instruments
are OTC (broker-synthesised prices). Splitting the pre-cutoff broker
walk-forward: spot 65.5% on 2,103 independent trades; OTC 47.1% on 136 -
below break-even AND below coin flip, which research contamination cannot
explain (bias inflates, it does not sink). The deployed meta's bucket slope
transfers on spot (61->72%) and only weakly on OTC (47->57%).

CONSEQUENCE: the edge is treated as SPOT-ONLY until forward evidence says
otherwise. The $1 execution track now places orders on spot signals only
(OTC signals are paper-logged with trade_skipped, so the frozen hypothesis
evaluation is untouched and still covers every instrument). A spot-only
policy variant is the natural candidate for the NEXT window's registration
if the forward OTC subset confirms this finding.

## SPREAD HAIRCUT — the decisive execution finding (2026-07-24, research_spread.py)

Dukascopy separate BID/ASK m1 candles, 2023-2025, standard pipeline on MID,
same trades rescored under three settlement rules. Median spread at traded
bars: 0.7 pips.

| meta gate | mid | half-spread | adverse |
|---|---|---|---|
| 0.60 | 58.5% | 45.1% | 35.5% |
| 0.65 | 59.4% | 45.3% | 34.8% |
| 0.70 | 61.9% | 46.0% | 34.1% |

MID reproduces the familiar leak-free numbers, confirming the method. The
alarming part: a mere 0.35-pip half-spread requirement drops the win rate
12-15 points, below the 53.5% break-even AND below a coin flip. This proves
the model's directional edge lives in SUB-PIP moves at the 15-minute
horizon - it is extraordinarily fragile to any execution friction.

FRAMING CORRECTION (recorded honestly): research_spread.py's docstring said
the truth "sits between mid and adverse". That is WRONG for binary options.
"Adverse" (enter ask, settle bid) models a SPOT/CFD trade that crosses the
FULL bid-ask on both legs. A binary option settles on expiry-price vs
strike-price measured on the broker's SINGLE quoted feed, so it does not
cross the full spread. The plausible binary range is mid-to-HALF, and even
the half-spread end is unprofitable.

CONSEQUENCE - this reorders the whole project's priorities:
- The forward-test candle win rate measures the MID column only; it is no
  longer the decisive number.
- The DECISIVE number is IQ's actual binary strike mechanic: does it strike
  and settle at mid (profitable), or apply any spread (fatal)?
- Only the $1 PRACTICE demo trial can measure it - IQ's check_win_v4
  returns the broker's real win/loss, and comparing that to our candle-mid
  label is a direct measurement of IQ's effective haircut. The demo trial's
  LABEL-FIDELITY (broker outcome vs candle label agreement) is now the
  single most important measurement in the project, above the forward
  window itself.

This is precisely the failure mode that killed the comparable public
projects (Farxida's 86x paper-vs-backtest collapse). ATLAS has now
QUANTIFIED the risk offline before spending real money, and pinned the one
live measurement that resolves it.

## FEED IDENTIFICATION — mid confirmed historically (2026-07-24)

The spread finding's decisive unknown ("which price is IQ's feed?") was
answerable from PREVIOUS data after all: joining 62-63k minutes per pair of
IQ spot candles against Dukascopy bid/ask for the same minutes
(2026-05-22..07-23):

| pair | IQ vs MID (median, pips) | vs BID | vs ASK | median abs dev |
|---|---|---|---|---|
| EURUSD | +0.00 | +0.15 | -0.15 | 0.05 |
| GBPUSD | -0.05 | +0.30 | -0.40 | 0.05 |
| USDJPY | +0.00 | +0.25 | -0.20 | 0.05 |

IQ's single price line IS interbank mid to within ~0.05 pips - far below
the 0.15-0.35 pip half-spread that killed the win rate in the stress
columns. Binaries settle strike-vs-expiry on this feed, so the MID column
(57-62%) is the structurally correct central expectation. The collapse
scenario now requires ORDER-TIME strike manipulation (strike != displayed
quote), not feed spread - a much narrower residual.

Demo-trial burden accordingly reduced from "decide which column is
reality" to "confirm no order-time trickery": click-time strike vs our
next-bar-open assumption (second-level, roughly symmetric), possible
order-time markup, last-second settlement behaviour.

## Pre-verdict audit corrections (2026-07-24) - recorded before ANY verdict

A three-agent code audit (pipeline correctness / research statistics /
live execution) ran while the forward window was still verdict-free; all
evaluator corrections below therefore precede any result they could bias.

REGISTRATION RECONCILIATION: the original H3 registration fixed the
primary at meta_p >= 0.60; a later same-day note drifted it to 0.65. The
ORIGINAL registration stands - verdicts are issued only for H2 primary
(ev 0.03), H2 secondary (ev 0.04), H3 (meta 0.60) and H4 (extra-vol, ev
0.03), matching ALPHA = 0.05/4. Every other threshold (ev 0.02, meta
0.65/0.70/0.775) is REPORTED without a pass/fail verdict. H4, registered
but previously never evaluated, is now evaluated.

EVALUATOR CORRECTIONS (criteria intent unchanged): (1) frozen-model loading
now refuses any pickle whose training data extends past the cutoff - a
routine retrain would previously have silently scored the forward window
with a model trained on it; (2) a zero-variance cluster vector (e.g. a
100% win record) previously produced a nan p-value and verdict FAIL - it
now falls back to an exact binomial on cluster majorities; (3) EV
decisions in the candles track use the causal observed payout, as the
frozen configuration specified, not the 0.87 fallback.

STATISTICS CORRECTIONS: the acceptance PBO matrix previously spanned
selection-era blocks whose meta_p were the meta model's predictions on its
own training rows, biasing PBO toward 0; recomputed on holdout-only blocks
(out-of-sample for the meta) PBO remains 0.00 - the biased computation's
conclusion happened to survive honest math. The acceptance Brier check now
covers ALL holdout rows (0.24943, passes non-vacuously) rather than the
EV-gated tail. Screening p-values in research_pooled/research_meta now
test one-sided against the economic break-even instead of two-sided
against 0.5; in-sample p-values quoted earlier in this document were
computed under the old flattering null and should be discounted
accordingly (the forward criteria were always break-even-based and are
unaffected).

KNOWN LIMITATIONS accepted and documented rather than fixed: the vendored
library's abandoned timed-out threads can race reconnects (mitigated by
per-cycle failure bail + relaunch); settlement overrun past the hour can
delay the next runner start; experiments.py id assignment is racy under
concurrent training runs; validation does not reject misaligned bar
spacing (feeds are aligned in practice).

## SECOND pre-verdict audit (2026-07-24, Windows host) - findings recorded, NO criteria changed

A four-dimension adversarial code audit (leakage / verdict readiness /
config fidelity / data integrity, every finding attacked by an independent
skeptic) ran while the window was still verdict-free. 11 findings confirmed,
11 refuted. NOTHING below has been acted on: no model was rebuilt, no
threshold moved, no payout semantics changed. The record exists so that
whatever is decided later provably predates any result.

BLOCKING - the frozen models are contaminated and the evaluator refuses them:
- models/h2-20260722.pkl has data_end_ts 2026-07-22T05:04Z; the registered
  cutoff is 2026-07-22T00:00Z. It trained on the first 5.07h of the forward
  window. models/h4-20260722.pkl: 2026-07-22T17:32Z, 17.53h inside.
  (Verified by reading the pickles directly. This also contradicts the H2
  in-sample line above, "data ..2026-07-21": the row count matches the
  pickle exactly, 1,105,496, but the data end does not.)
- models/meta-h3.pkl carries NO data_end_ts, so the guard cannot check the
  PRIMARY hypothesis's model at all.
- load_bundle's cutoff guard (written by the first audit for exactly this)
  correctly refuses both. It sits on the first line of candles_track, so the
  documented verdict-day command aborted with one stderr line and no output.
  Its message advises "point at the pre-cutoff pickle explicitly" but no such
  option existed, no pre-cutoff pickle exists on disk or in git, and no code
  path can build one (research_pooled.load_pooled takes no cutoff argument).
- Contamination bound by elapsed time: ~1.5% of a 14-day window for H2,
  ~5.2% for H4.

EVALUATOR DEFECTS FOUND, DELIBERATELY NOT FIXED (both would make PASSING
EASIER, so they are the operator's call, not a maintainer's):
- observed_payout() reads spec.option_kind, which is 'turbo' for 24 of the
  tradable instruments, while the contract traded and the frozen config
  ("binary-kind payout") are BINARY. Measured on collected snapshots: binary
  pays +0.0223 over turbo on average, moving break-even from 0.5412 (as
  coded) to 0.5348 (as registered) - 0.64pt on a test whose margin is a few
  points.
- MIN_CLUSTERS_CANDLES = 30 while H2/H3/H4 are registered at ">= 20
  clusters" (the >= 30 in the Protocol section belongs to H1). As coded the
  candles track is stricter than registered.
- paper_track applies no frozen-universe filter, so its registered verdicts
  would include post-registration instruments (candles_track does filter).

LIVE-EXECUTION FIDELITY (affects signals already logged):
- The runner computes cross-asset currency-strength features over the live
  39-instrument basket; the frozen model was trained on 16. Measured shift
  in xs_mkt_vol: 42%. Every signal logged since 2026-07-24 carries
  out-of-distribution cross-asset columns.
- live_h2_runner's model loader has no post-cutoff refusal, so a retrain
  would silently swap a contaminated model into the live window.

CHANGES ACTUALLY MADE (criteria-neutral, ergonomics only):
- forward_eval gained --h2-model/--h4-model globs, implementing the remedy
  the refusal message already advertised. The post-cutoff guard still
  applies to whatever bundle is selected; it cannot be bypassed.
- A candles-track SystemExit no longer takes the leak-immune paper track
  down with it; the failure is reported inside the JSON report instead.

OPEN DECISIONS (owner: operator):
1. The contaminated models. Options considered: (a) move the registered
   cutoff to 2026-07-22T17:32Z - excludes every contaminated hour, touches
   no artifact, costs ~5% of the window, moves the boundary in the
   conservative direction; (b) rebuild H2/H4 strictly pre-cutoff - needs a
   ~60-day backfill and an until_ts on load_pooled, and yields a model that
   is NOT the one that generated the live signals; (c) proceed with
   disclosure.
2. Whether to correct the payout-kind and cluster-threshold defects above.
   Both loosen the test, so correcting them after results exist would be
   indefensible; correcting them now, before any verdict, is the only
   honest window - or they stand as coded and the verdict is read with this
   record attached.

## Notes

- Real execution frictions (spread at entry, expiry timing, requotes) are
  NOT modeled; a pass here justifies a PRACTICE-balance live trial next, not
  real money.
- Trades concentrated in later folds in-sample (model needed ≥~400k rows
  before clearing the EV gate); expect low trade counts early in the window.
