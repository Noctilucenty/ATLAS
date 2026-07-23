# CLAUDE.md — ATLAS project context (auto-loaded)

Read `FINDINGS.md` (complete history: every experiment, number, rejection,
bug, lesson) and `FORWARD_TEST.md` (pre-registered protocol — GOVERNS) at
the start of any substantive session. This file is the compressed state.

## What this is

IQ Option binary-options research: MCP server (`server.py`) + walk-forward
ML pipeline. LightGBM direction model + meta-labeling filter + EV gating
against live payouts. Demo (PRACTICE) only; `IQ_ALLOW_REAL=0` stays 0.

## Current state (2026-07-24)

- **Edge is real, calibrated expectation 57–65% WR, SPOT ONLY** (OTC is
  47% — below coin flip; demo trading skips OTC). Break-even 53.5%.
- Modern holdout 67–81% by meta threshold (0.60→0.775) is partly
  era/research-inflated; untouched 2003–15 era says ~57%.
- **Edge is sub-pip**: half-spread friction kills it (45%). Saved by feed
  identification: **IQ's feed IS interbank mid (±0.05 pips)** → MID
  settlement column is the correct expectation.
- Frozen models: `models/h2-20260722.pkl`, `h4-20260722.pkl`,
  `meta-h3.pkl`. The eval loader REFUSES post-cutoff-trained pickles.
- Awaiting: forward verdicts (~Jul 28–Aug 6) + demo-trial label fidelity
  (broker verdict vs candle label over ~100 trades = IQ's order-time
  behaviour — the single most important pending measurement).

## Hard rules (violations wreck the project)

1. **Never modify pre-registered hypotheses/criteria** in FORWARD_TEST.md
   after forward data exists. Verdicts: run `forward_eval.py` ONCE.
   Registered verdict set = {H2p ev0.03, H2s ev0.04, H3 meta0.60, H4},
   α = 0.05/4. Everything else is reported, never pass/failed.
2. **Registry discipline**: every experiment family →
   `registry.record(...)` in `research_registry.jsonl` (~154 trials).
   Deflation uses this count; an unrecorded experiment is a lie.
3. **Leak discipline**: selection ≤2022 / holdout ≥2023; meta models fit
   on selection only; time-purged folds (purge = horizon); per-asset
   independent de-overlap + cross-asset clusters for significance; raw
   overlapping-trade counts are meaningless.
4. **Do not re-try the graveyard** (numbers in FINDINGS.md §Phase 4):
   tuning, HAR-RV, ensembles-for-WR, meta-v2, consensus, isotonic,
   day-of-week, hour blocklists, NFP blackout, second meta, specialists,
   triple-barrier, deep learning. The meta filter already absorbs
   calendar/session/event structure.
5. **Git**: commit verified work straight to main, push, NO Claude
   attribution ever. Verify repo stays PRIVATE via API when touching it.
6. **Verify before claiming** — check API/state, don't assume (past
   failures: "private" repo that was public; "resolved" with OTC blind
   spot; vacuous tests that asserted nothing).

## Operational map

- Location `~/dev/ATLAS/ATLAS` (moved off ~/Desktop: TCC blocks launchd
  there). Agents: `com.atlas.iqoption-collector` (hourly :05, no
  KeepAlive by design) and `com.atlas.h2-paper` (hourly :00, --trade,
  KeepAlive+throttle, flock single-instance). Self-healing:
  `atlas_hook.zsh` in ~/.zshrc (catchup + agent watch + notifications);
  `catchup.sh` backfills (broker keeps ~60d). `status.sh` = dashboard.
- Broker quirks: OTC volume=0; per-table instrument keys
  (`instruments.py`); `get_all_open_time` broken (freshness proxy);
  server reads .env at start only; payout-quoted ≠ candle-fetchable;
  binary payouts > turbo; USDCHF spot = data-only (no option market).
- Tests: `.venv/bin/python -m pytest -q` (133) — run before every commit.
