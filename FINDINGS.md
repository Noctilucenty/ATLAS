# ATLAS — Complete Findings Record

Every experiment, number, rejection, bug, and operational lesson from the
research program (2026-07-21 → 2026-07-24). This file is the exhaustive
record; `CLAUDE.md` is the short auto-loaded context that points here, and
`FORWARD_TEST.md` is the pre-registration/protocol document. When they
disagree, FORWARD_TEST.md governs protocol; this file governs history.

Registry discipline: every experiment family is appended to
`research_registry.jsonl` via `registry.record()` — **~154 variants** as of
2026-07-24. All quoted statistics are penalised against that count where
deflation applies.

---

## 1. The headline state (read this first)

- **The directional edge is real**: it survives two independent decades
  (2016–25 research era; 2003–15 fully untouched), three currency pairs,
  ~7.4M labeled rows, a 150+-trial deflation penalty (deflated z 5–12),
  PBO 0.00 (holdout-only CSCV), and Monte-Carlo-verified statistics.
- **Calibrated expectation: 57–65% win rate, spot instruments only**, vs a
  53.5% break-even at 0.87 payout. The 67–81% modern-era holdout numbers
  are partly era-specific/research-inflated (the untouched era says ~57%).
- **The edge is SUB-PIP and execution-fragile**: a 0.35-pip half-spread
  requirement collapses the win rate 12–15 points to below coin-flip. BUT
  **IQ's price feed is interbank mid within ~0.05 pips** (measured over
  62–63k minutes/pair), and binaries settle on that single feed — so the
  profitable MID column is the structurally correct expectation.
- **OTC instruments are dead weight**: 47.1% (below coin flip) on 136
  independent trades vs spot 65.5% — the broker's synthetic books resist
  the model. Demo trading is spot-only.
- **The single lever that moves win rate is the meta-filter threshold.**
  Everything else tested — 12+ ideas — either failed or was already
  absorbed by the meta filter.
- Two live measurements remain: the pre-registered **forward test** (first
  verdicts ~2026-07-28) and the **$1 demo trial's label fidelity** (broker
  verdict vs candle-mid label ≈ IQ's order-time behaviour; ~100 trades).

---

## 2. Chronology with every result

### Phase 1 — Setup and data (2026-07-21/22)

- Repo `Noctilucenty/ATLAS` cloned to `~/Desktop/dev/ATLAS/ATLAS` (later
  moved, see §7). `iqoption-mcp` repo was byte-identical → deleted.
- Python 3.12.13 via uv; vendored `iqoptionapi` cloned+installed; MCP
  server registered as `iqoption` (user scope).
- **Broker quirks discovered**: `.env` read only at server start (reconnect
  via /mcp after edits); login "invalid_credentials" until server restart
  picked up new credentials; ~**60 days** 1-minute candle retention; **OTC
  candles report volume=0 on every bar** (crashed the feature pipeline —
  fixed with neutral `vol_rel=1.0`, FEATURE_VERSION 1.2.0);
  `get_all_open_time` crashes inside the vendored lib (openness inferred
  from candle freshness instead); broker uses DIFFERENT keys per table
  (candles vs payout vs order — `instruments.py` binds all three);
  **payout presence ≠ fetchable candles** (AUDUSD-OTC quotes payouts but
  is absent from the vendored ACTIVES map → EURGBP-OTC substituted);
  USDJPY-OTC has binary-only quotes; USDCHF spot has candles but NO option
  market (`tradable=False`, kept for CHF-strength context);
  **binary-kind payouts (0.87–0.88) beat turbo (0.82–0.86)** → break-even
  53.2–53.5% instead of ~54.6%.
- Backfills: EURUSD + EURUSD-OTC ~149k candles (broker only serves ~60d;
  data starts 2026-05-21). Later 16 instruments (1.26M), finally 28
  instruments (~2.2M candles) + payout snapshots (10k+).

### Phase 2 — First models and the replication lesson

- **Broker sweep, 16 variants** (EURUSD/EURUSD-OTC × logreg/lgbm ×
  h=5/15/30/60): best cell EURUSD lgbm h15 — raw 579 trades 58.5%
  "p<0.001" which collapsed to **93 independent trades, 60.2%, p=0.061**
  after overlap correction. OTC variants ≈ nothing.
- **Overlap lesson (permanent)**: consecutive 1-min signals share up to
  h-1 bars of one forward window; raw-trade binomials are fake. All
  evaluation since uses per-asset independent de-overlap AND cross-asset
  chain clusters.
- The 60.2% **died under perturbation**: Optuna-tuned 55.4%/65 (tuner fled
  to minimum capacity on every fold — diagnostic of no signal);
  v1.3-features run 51.6%/31. Verdict: best-of-16 selection noise.
- Features v1.3.0 added: adx, bb_pctb, macd_hist_atr, ret_60.

### Phase 3 — Pooling breakthrough and the hypothesis family

- **Pooled 10-instrument walk-forward** (time-purged folds — with ~10
  assets interleaved per minute, row-based gaps under-purge): 462 raw →
  130 independent → **63.9%, p=0.002**; 70 cross-asset clusters, mean frac
  67.5% (p=0.0007), 47W/22L majorities, spread over 17 dates; all 5 fold
  Briers < 0.250. → `FORWARD_TEST.md` created, H1/H2 registered
  2026-07-22 with frozen configs and pre-committed criteria.
- **H2** (16 instruments, cross-asset currency-strength features
  xs_base_str/xs_quote_str/xs_mkt_vol, realistic entry_next_open labels):
  staircase by EV gate 66.7% → 75.0% → 80.5% → 90.0% (independent trades
  105/64/41/20). Monotone gate→WR = calibration signature.
- **Deep-history anchors** (histdata.com 1-min, EST→UTC fixed −5): EURUSD
  2016–25: 19,016 ind trades 56.9→58.6% by gate, mean Brier 0.24939, all
  8 folds < 0.25. GBPUSD: 22,202 @ 56.4→58.9%, 0.24935. USDJPY: 14,638 @
  55.7→58.1%, 0.24940. ~7.4M rows, identical signature.
- **Meta-labeling (H3)**: LGBM on 42,619 selection trades (2016–22),
  honest holdout 2023–25 (13,237 trades, base 55.9%): 0.55→57.4% (66%
  kept), 0.60→**61.9%** (25% kept), 0.65→68.2% (10%). Top meta features:
  hour_sin/cos, adx, atr_norm, conf, vol_regime. **Late-UTC session
  (21–24h) replicated**: 62.8% selection → 60.7% holdout vs ~54% others.
  ADX buckets did NOT replicate (flat→weak-favoured flip) — never used.
- **Deployed artifacts**: `models/h2-20260722.pkl` (1.105M rows, 16
  instruments), `models/meta-h3.pkl` (refit on all 55,856 decade trades),
  later `models/h4-20260722.pkl` (extra-vol block).

### Phase 4 — The lever hunt (what worked and the graveyard)

**Survived:**
- **extra_vol block** (gk_vol, rs_vol, park_vol, cs_spread, vol_of_vol):
  +0.5..+1.5pt on **9/9** pair×gate decade comparisons (mean +0.7);
  counter-evidence recorded: cluster fraction −0.3..−1.4 on all pairs.
  → registered as **H4**, never retrofitted.
- **Meta threshold dial** (the one true lever): pooled-global holdout
  61.2% @0.60 → 67.4/69.5 @0.65 → 71.5/73.0 @0.70 → 75.8 @0.725 → 77.2
  @0.75 → **80.9% @0.775 (392 ind trades)** → 0.80 thin (57).
- **Global > specialists**: 3/3 thresholds (62.7/69.5/73.0 vs
  61.2/67.4/71.5), usually MORE trades.
- **Breadth scales volume not ceiling**: 9-pair pooled decade = same
  staircase (80.6 vs 80.9 @0.775) at **2.7×** trades.
- **Horizon**: h10/15/20/30 ≈ 58.3/58.6/58.5/57.8% at tight gate — h15
  confirmed near-optimal; more independent bets at shorter horizons.

**Rejected (with numbers — do not re-try):**
- Optuna tuning: 55.4% vs 60.2 untuned; chose minimum capacity every fold.
- HAR-RV: first run INVALID (1-day RV window + median-24-bar segments
  discarded 85% of rows — trained on 530k vs 2.48M); fixed 15m/1h/4h
  windows: wash on WR (57.1/57.2/58.8 vs 56.9/57.6/58.6), worse clusters
  (65.2 vs 67.5), worse Brier, 43% fewer trades. Volatility predicts
  magnitude, not direction.
- Ensemble (3-seed LGBM + logreg averaging): WR −0.3..+0.4 (no gain);
  cluster frac +2.6 and **+25% trades** — filed as statistical-power tool
  if the forward window runs thin, not a WR lever.
- Meta v2 (enriched context + p_up_ext + causal per-asset streak_20/50):
  78.2% vs v1's 79.1% @0.775 — no gain.
- Consensus gate (base & enriched models agree): **byte-identical** trade
  sets at every threshold — high-meta survivors already agree.
- Per-pair isotonic recalibration: rescales the threshold axis, no
  selection gain at matched volume (would matter for stake sizing only).
- Day-of-week (dow_sin/cos): −0.4/+1.6/−0.7/+0.6pt across gates —
  alternating-sign wash; hour+meta absorb calendar structure.
- Explicit hour blocklists / second rejection model / NFP blackout: all
  **redundant with the meta filter** — e.g. near-NFP (first Friday, 08:30
  NY, DST-correct) ungated trades run 39–44% (toxic!) but meta≥0.65 takes
  **zero** trades in ±60min of NFP across 3 holdout years.
- Triple-barrier labels: wrong product (binaries are fixed-expiry).
- Deep learning / more indicator dumping / regime-specialist models: not
  pursued (scale wrong; specialists already lost; curated-features
  philosophy validated by `vol_rel` importance literally 0 in every fold).
- Weak features identified (stability across all folds): vol_rel (0),
  wick/body ratios, session one-hots — dominated by hour_sin/cos,
  atr_norm, vol_regime, ret_60, adx.

### Phase 5 — Honesty infrastructure

- **Leak found & fixed in own tooling**: research_wr originally scored
  holdout with the deployed meta (trained on ALL years) — inflated
  baseline 65% → true leak-free 61.2%. All meta scoring since trains on
  selection only.
- **Acceptance contract** (`acceptance_report.py`, 7 fixed checks; paper
  check settable ONLY by the forward test): holdout edge @ Bonferroni,
  PBO < 0.40 (CSCV), deflated win rate vs honest trial count, Brier <
  0.25 (ALL holdout rows), meta ECE < 0.05, ≥200 ind trades. Global
  candidate (ev 0.03, meta 0.65): PROVISIONAL PASS — 69.5% on 1,981;
  0.775 candidate: 80.9% on 392, deflated z 8.2. Meta ECE 0.023 overall;
  **USDJPY runs 2–3pt overconfident** (known, within gate).
- **MinTRL** (Bonferroni α=0.0125): 598 trades needed at true 57%, 163 @
  62%, 72 @ 66%, 39 @ 70%, ~13 @ 79%.
- **Monte-Carlo verification of the referee**: expected_max_z
  conservative by 0.02–0.04; deflated null false-pass **0.2%** (target
  5%; undeflated would be 92%); PBO noise-calibrated 0.494 over 300
  matrices — but single-reading sd ±0.20 at 8 folds (a lone PBO is "low",
  not "precisely zero").
- **Era holdout 2003–2015** (pre-registered BEFORE download; 13 untouched
  years): REPRODUCED per pre-committed criteria — 56.7/57.1/57.1% at
  fixed thresholds 0.65/0.70/0.775 on 14,468/9,375/3,204 ind trades,
  worst p=2e-5 — but staircase rise only 0.4pt vs 12pt modern.
  **Expectation anchored 57–65%, not 79%.**
- **Calibration of p_up**: predicted 0.509/0.530/0.567 vs actual
  0.509/0.529/0.572 — near-perfect; EV gate operates on trustworthy
  probabilities (slightly under-confident at extremes = conservative).

### Phase 6 — Segments, events, execution reality

- **OTC split (pre-cutoff broker window)**: overall 64.4% (2,239 ind) —
  spot **65.5%** (2,103) vs OTC **47.1%** (136, below coin flip;
  contamination inflates, never sinks → finding is real). Meta bucket
  slope transfers on spot (61→72%) incl. 25 never-trained pairs; weak on
  OTC (47→57%). → **demo trading restricted to spot** (`trade_skipped`
  on OTC signals; paper/hypothesis coverage unchanged).
- **Spread study** (Dukascopy separate bid/ask m1, 2023–25, meta ≤2024-06
  scored after): MID reproduces 57.3–61.9% (method confirmed);
  **half-spread → 45.1–46.0%; adverse → 34.1–37.0%**. Median spread at
  traded bars 0.7 pips → **the edge is SUB-PIP**. Framing correction:
  "adverse" models a SPOT trade crossing full spread; binaries settle
  expiry-vs-strike on ONE feed → plausible binary range is mid-to-HALF —
  and even half is below break-even.
- **Feed identification (the de-risking)**: IQ candle closes vs Dukascopy
  same-minute bid/ask over 62–63k minutes/pair (2026-05-22..07-23):
  IQ−MID median +0.00/−0.05/+0.00 pips (EURUSD/GBPUSD/USDJPY), median
  |dev| 0.05 pips, squarely between bid (+0.15..+0.30) and ask
  (−0.15..−0.40). **IQ's single feed IS interbank mid** → the MID column
  is the structurally correct expectation. Residual unknowns (demo trial
  only): click-time strike vs next-bar-open timing, order-time markup,
  last-second settlement.

### Phase 7 — The code audit (2026-07-24, three parallel reviewers, pre-verdict)

All fixes applied BEFORE any verdict existed (which is what makes
evaluator corrections legitimate). 133 tests green after.

Critical fixes:
- **Cutoff guard**: `forward_eval.load_bundle` now refuses any model pickle
  whose `data_end_ts` > cutoff — a routine retrain would have silently
  trained on the forward window and invalidated the test.
- **nan verdict bug**: zero-variance cluster vector (e.g. 100% wins) →
  t-test nan → verdict FAIL. Now exact binomial on cluster majorities.
- **Verdict/alpha drift**: code issued ~11 verdicts under an α=0.05/4
  claim; H4 (registered) was never evaluated; the H3 primary had drifted
  0.60→0.65. Resolved: verdicts ONLY for {H2p ev0.03, H2s ev0.04,
  H3@0.60 (ORIGINAL registration stands), H4}; 0.02/0.65/0.70/0.775
  reported without verdicts; H4 now evaluated.
- **PBO contamination**: acceptance matrix had selection-era blocks scored
  by the meta on its own training rows (PBO→0 by construction).
  Recomputed holdout-only: **still 0.00** — conclusion survived honest
  math. Brier check moved to ALL holdout rows (0.24943, non-vacuous).
- **Screening nulls**: research_pooled/meta tested two-sided vs 0.5;
  now one-sided vs economic break-even. Historic in-sample p-values in
  FORWARD_TEST.md were computed under the flattering null — discount them
  (forward criteria were always break-even-based).
- **Live runner**: dead-connection cycles previously skipped both the
  heartbeat AND the failure counter (a connection dying at minute 3
  burned the slot and exited 0 — the relaunch design defeated by its own
  empty-frame path); flock single-instance lock (launchd + terminal +
  hook could double-run → duplicate signals/orders); settlement on every
  exit path; per-asset freshness timestamps (one stale `now` mis-labelled
  tail assets after early timeouts); binary-payout `or`-falsy fallthrough
  to turbo payout fixed; plist ThrottleInterval 120s (crash-loop was
  hammering login every ~10s); hook kickstart threshold 10min→75min.
- **Pipeline**: the suite's ONLY label-direction and tie tests were
  vacuous (`.all()` on empty frames — an inverted label would have passed
  everything); rebuilt with warmup-surviving fixtures, both directions.
  `_mtf_context_features` fabricated 0.5 on warmup rows → NaN. Storage
  merge now flags NaN-vs-value dataset disagreements.
- Verified clean: causal features (labels cross-checked row-by-row against
  raw closes, both entry modes), purge arithmetic everywhere, balance-mode
  guard (verified to the wire: reconnect CANNOT flip PRACTICE→REAL),
  JSONL append atomicity, `meta_probability` row alignment, `settled`
  record shape.
- Known limitations (accepted, documented): vendored lib's abandoned
  timed-out threads can race `connect()` (mitigated by fail-bail+relaunch);
  settlement overrun can delay next hourly start; `experiments.py` id
  assignment racy under concurrent runs; `validation.py` doesn't reject
  misaligned bar spacing.

---

## 3. Operational history & infrastructure

- **TCC saga**: `~/Desktop` is macOS-protected; launchd agents failed
  ("can't open input file", later exit 78 with NO output after reboots)
  even after Full Disk Access grants. Manual runs always worked
  (terminal has access). **Fix: project moved to `~/dev/ATLAS/ATLAS`
  (2026-07-23)** — venv survived (uv symlinks), MCP re-registered, plists
  rewritten; both agents PROVEN running under launchd from the new home.
- **Agents**: `com.atlas.iqoption-collector` (hourly :05; NO KeepAlive —
  its nonzero exits are a designed health signal; auto-relaunch would
  hammer the broker through weekend closures) and `com.atlas.h2-paper`
  (hourly :00, 57-min runs, KeepAlive on failure-exit + ThrottleInterval
  120s, `--trade` mode).
- **Self-healing**: broker serves ~60d history on demand → `catchup.sh`
  (gap-aware, registry-derived asset list, success-gated marker) +
  `atlas_hook.zsh` in `~/.zshrc` (auto-catchup ≥6h stale, agent watch
  ≥75min stale heartbeat → kickstart + macOS notification). Sleep loses
  only demo-trade minutes and payout snapshots; candles/forward test
  self-heal. launchd re-fires missed calendar jobs on wake.
- **Runner logging** (`logs/live_h2.jsonl` + `live_h2_heartbeat.jsonl`):
  per signal — p_up, meta_p, h4_p (shadow), payout at decision, true
  wall-clock ts, decision_latency_s, mode, `trade_skipped` (OTC),
  order_id, then a separate `settled: true` record with broker
  result/profit. Heartbeat every cycle incl. empty-frame failures.
  Derivable shadow tracks at scoring time: no-meta (=H2 primary), every
  meta threshold, INVERTED (flip actions), no-model baseline (=breakeven).
- **Repo hygiene**: GitHub `Noctilucenty/ATLAS` is PRIVATE (was
  accidentally PUBLIC ~1h — `gh repo create --private` partially failed
  and silently dropped the flag; **verify via API, never assume**); repo
  was deleted+recreated once (external deletion; all 28 commits restored
  from local); history rewritten to strip 9 `Co-Authored-By: Claude`
  trailers — **never add Claude attribution**; commit verified work
  straight to main and push (user rule); no LICENSE = all rights
  reserved (intentional).
- **First live events**: first paper signal 2026-07-22 (GBPUSD call,
  meta 0.65) — LOST (1 trade, meaningless); 4 EURUSD puts fired
  2026-07-23 21:55–58Z ~10min before trade mode went live (paper-only).
  Signals arrive in bursts (consecutive minutes, same asset/direction =
  ONE independent bet). 0 demo orders placed as of 2026-07-24.

---

## 4. Current pending items

1. **Forward test** (`forward_eval.py`, run ONCE): H2p verdict reachable
   ~2026-07-28 (~6 clusters/day), H3@0.60 ~Aug 1, full family ~Aug 6.
   Candles track = MID-settlement by construction (post-spread-finding it
   measures direction quality, not executability).
2. **Demo trial label fidelity** — NOW THE KEY MEASUREMENT: broker
   `check_win_v4` verdict vs candle-mid label agreement over ~100 spot
   trades ≈ IQ's order-time haircut. Requires the Mac awake when signals
   fire (bounded sessions fine; ~6 spot signals/day).
3. **PocketOption lane (parked)**: account created; OTC there is also
   broker-synthetic → forward-collection only (no history exists);
   needs user's session ID for `BinaryOptionsTools-v2`; weeks-scale.
4. If forward passes + fidelity high → the 0.775 operating point
   (80.9% modern / ~57% era floor) is the deployment candidate — via a
   NEW registration, never by editing the frozen ones.
