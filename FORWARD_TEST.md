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

## Notes

- Real execution frictions (spread at entry, expiry timing, requotes) are
  NOT modeled; a pass here justifies a PRACTICE-balance live trial next, not
  real money.
- Trades concentrated in later folds in-sample (model needed ≥~400k rows
  before clearing the EV gate); expect low trade counts early in the window.
