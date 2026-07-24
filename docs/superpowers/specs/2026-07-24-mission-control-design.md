# ATLAS Mission Control — design (2026-07-24)

Approved by user 2026-07-23 (US) / 2026-07-24 UTC. Scope: Windows always-on host `C:\ATLAS`.

## Constraints (non-negotiable)
- Read-only against all trading state (journal.db, market.duckdb via `read_only=True`, jsonl logs).
- Never evaluates pre-registered forward-test criteria (`forward_eval.py` runs ONCE, by hand, later).
- No changes to the signal path, models, FORWARD_TEST.md, or supervisor/runner.
- New modules get pytest coverage; suite must stay green.

## Components
1. **status.py** — one-shot terminal status: scheduled-task state, pids, heartbeat age,
   last collect, DB freshness, trade counts, payout age. Exit code 0 healthy / 1 warning / 2 critical
   (scriptable).
2. **dashboard.py** — stdlib-only localhost web dashboard (default port 8787), auto-refresh.
   Panels: forward-test progress (counts only), label-fidelity tracker (broker verdict vs
   candle label agreement), WR by asset/hour/conf bucket, calibration, equity curve,
   payout landscape, three-tier health banner. Self-contained HTML/JS (no CDN).
3. **watchdog.py** — every 15 min via Task Scheduler (interactive, `pythonw.exe`, no window):
   heartbeat stale >10 min, task not running, or logs frozen → Windows toast. Optional Telegram later.
4. **RESEARCH_QUEUE.md** — post-verdict experiment queue (registered-discipline compatible).

## Data sources
`logs/live_h2_heartbeat.jsonl`, `logs/live_h2.jsonl`, `logs/supervisor.log`, `journal.db` (sqlite),
`market.duckdb` (read-only, freshness + payouts), `models/*.pkl` (names only).

## Out of scope
Model changes, new signals, auto-evaluation of hypotheses, anything on the Mac.
