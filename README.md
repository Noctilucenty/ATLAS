# ATLAS

**A research pipeline for short-horizon FX binary-option prediction on IQ Option — with a bias-resistant validation harness that tries very hard to prove its own edge is fake.**

ATLAS pairs an MCP server (so Claude can read the broker and place demo trades) with a walk-forward machine-learning pipeline: collect 1-minute candles across dozens of currency pairs, engineer a curated feature set, train calibrated gradient-boosted models, and decide trades by **expected value** against the live payout. Its defining feature is not the model — it is the discipline around it. Every result is checked against overlapping-trade inflation, cross-asset correlation, a decade of out-of-sample data, and a **pre-registered forward test** whose success criteria were frozen before any forward data existed.

> ### ⚠️ Read this first
> - **Unofficial API.** IQ Option has no public API. The [`iqoptionapi`](https://github.com/iqoptionapi/iqoptionapi) library is reverse-engineered: logins can break without notice, automated trading may violate IQ Option's Terms of Service, and accounts using it can in principle be flagged.
> - **Demo only, by default and by design.** Live trading is disabled unless you explicitly set `IQ_ALLOW_REAL=1`. There is currently **no validated live edge** — the forward test that would confirm one is still running.
> - **This is research, not financial advice.** Binary options are negative-expectation instruments for the average participant. Nothing here is a promise of profit.

---

## Why this project is unusual

Most retail "trading bot" repositories report incredible win rates because they fool themselves — they test on data the model has seen, count overlapping trades as independent, or tune until the backtest looks good. ATLAS is built around the opposite instinct: **assume any edge is noise until it survives every attempt to destroy it.**

Concretely, that means:

| Guardrail | What it prevents |
|---|---|
| **Chronologically purged walk-forward** | Training on the future; label leakage across the train/test boundary |
| **Probability calibration** | Confident-but-wrong models; the decision rule needs *true* probabilities, not just rankings |
| **Independent-trade & cross-asset clustering** | Counting one correlated burst of trades as many wins |
| **Decade-scale replication** on external data | Mistaking a two-month fluke for a real effect |
| **Pre-registration** (`FORWARD_TEST.md`) | Moving the goalposts after seeing results |
| **Bonferroni correction** | Declaring victory because *one of several* hypotheses passed by luck |

The honest scoreboard lives in [`FORWARD_TEST.md`](FORWARD_TEST.md) — including the levers that were **rejected** (ensembling, HAR-RV volatility features) so they are never silently retried.

---

## How the strategy decides to trade

Every minute, for each registered instrument:

1. **Model → probability.** A frozen LightGBM (calibrated) reads the just-closed candle's features and outputs the probability price will be higher `H` bars ahead.
2. **Expected-value gate.** It does not ask "are we confident?" — it asks "does the bet pay?" For a call at payout `r`:
   ```
   EV = p_up · r − (1 − p_up)
   ```
   A trade fires only if `EV` beats a margin. Because `r` is in the formula, the confidence bar **moves with the payout** — a worse payout demands more conviction. Puts are symmetric (fired when `p_up` is low).
3. **Meta-filter (quality layer).** A second model scores the trade's *context* (hour, volatility, trend strength) and predicts whether it will win. Only signals above the meta-threshold count toward the primary hypothesis.

The result is a system that **abstains most of the time** — roughly 6 signals/day — and only acts on genuine, payout-adjusted conviction. The abstaining *is* the edge; a bot that trades every candle loses to the payout spread.

---

## Repository layout

### Core pipeline
| File | Role |
|---|---|
| `server.py` | MCP server exposing `iq_*` tools to Claude |
| `instruments.py` | Broker instrument registry (28 instruments; per-asset candle/quote/order keys, verified live) |
| `collector.py` | Historical 1-minute candle + payout-snapshot collector |
| `storage.py` | DuckDB store with canonical, deduplicated, gap-aware history |
| `features.py` | Versioned, leakage-safe feature/label pipeline |
| `train.py` | Walk-forward train-freeze-predict orchestrator with calibration |
| `analyzer.py` · `execution_guard.py` | Deterministic EV signal policy; hard contract/`PRACTICE` guards |

### Research (screening only — never feeds execution)
| File | Role |
|---|---|
| `research_pooled.py` | Pooled cross-asset walk-forward + cross-asset currency-strength features |
| `research_deephistory.py` | Decade-scale anchor on free [histdata.com](https://www.histdata.com) 1-minute bars |
| `research_meta.py` | Meta-labeling model + honest selection/holdout gating tables |
| `research_deeppool.py` | Pooled decade run with cross-asset feature ablation |

### Forward test (the referee)
| File | Role |
|---|---|
| `FORWARD_TEST.md` | Pre-registered hypotheses, frozen configs, success criteria |
| `live_model_build.py` | Freeze a model to `models/*.pkl` with full provenance |
| `live_h2_runner.py` | Live **paper** runner (PRACTICE-guarded; `--trade` opt-in) |
| `forward_eval.py` | Runs the pre-registered test **once**, candles + paper tracks |

### Operations
| File | Role |
|---|---|
| `catchup.sh` | Gap-aware retroactive backfill (candles recover ~60 days on demand) |
| `run_both.sh` · `run_collector_loop.sh` · `run_paper_loop.sh` | Terminal-driven collection/paper sessions |
| `status.sh` | One-glance dashboard: agents, data freshness, signals |
| `atlas_hook.zsh` | Optional shell hook: self-heals stale data on terminal open |

---

## Setup

Requires **Python ≥ 3.12** and the vendored, reverse-engineered API library.

```bash
# 1. Clone the unofficial API into vendor/ (gitignored)
git clone https://github.com/iqoptionapi/iqoptionapi vendor/iqoptionapi

# 2. Create the environment and install dependencies
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    mcp duckdb pandas pandera pyarrow ta scikit-learn lightgbm optuna pytest \
    ./vendor/iqoptionapi

# 3. Add credentials
cp .env.example .env        # then fill in IQ_EMAIL / IQ_PASSWORD

# 4. Verify
.venv/bin/python -m pytest -q         # 120 tests
```

Register the MCP server with Claude Code (adjust the path to your checkout):

```bash
claude mcp add --scope user iqoption -- \
  /absolute/path/to/ATLAS/.venv/bin/python \
  /absolute/path/to/ATLAS/server.py
```

### Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `IQ_EMAIL` / `IQ_PASSWORD` | — | Broker credentials (never committed) |
| `IQ_DEFAULT_BALANCE` | `PRACTICE` | `PRACTICE` or `REAL` |
| `IQ_ALLOW_REAL` | `0` | Trading tools refuse the REAL balance unless this is `1` |

> **Note:** `server.py` reads `.env` only at process start. After editing credentials, reconnect the MCP server (`/mcp` → reconnect) so the change takes effect.

---

## Quick start

```bash
# Collect two months of history for all registered instruments
.venv/bin/python collector.py candles $(.venv/bin/python -c \
  "from instruments import INSTRUMENTS; print(' '.join(INSTRUMENTS))") \
  --interval 60 --hours 1440

# Screen the strategy on ten years of free spot data (no broker needed)
.venv/bin/python research_deephistory.py --pair eurusd --entry-next-open

# Keep the dataset current without an always-on process
./catchup.sh                         # backfills only the missing gap
```

---

## MCP tools

The server exposes IQ Option to Claude with a hard PRACTICE-only guard on every trading tool.

| Tool | Purpose |
|---|---|
| `iq_connect` / `iq_status` | Connect (handles SMS 2FA); connection & balance status |
| `iq_switch_balance` / `iq_reset_practice_balance` | Switch PRACTICE ↔ REAL; refill demo balance |
| `iq_find_asset` / `iq_get_candles` | Search assets; historical OHLC candles |
| `iq_open_assets` / `iq_payouts` / `iq_instruments` | Market openness; payout ratios; instrument ids |
| `iq_positions` | Open positions per instrument type |
| `iq_place_binary` / `iq_binary_result` | Place a call/put; await its win/lose outcome |
| `iq_place_order` / `iq_close_position` / `iq_cancel_order` | Margin orders with TP/SL; close; cancel |

---

## Design notes & known quirks

- **The broker uses different keys for the same instrument** in different tables (candles vs. payout vs. order). `instruments.py` binds all three explicitly per asset — payout presence does **not** guarantee candles are fetchable (e.g. `AUDUSD-OTC`).
- **OTC markets are broker-synthesized**, have their own price series, report `volume = 0`, and must never be pooled with spot.
- **`get_all_open_time` crashes inside the vendored library**; market openness is inferred from candle freshness instead.
- **Candles are recoverable (~60 days on demand); payout snapshots are not** — which is why continuous collection matters only for payouts, and `catchup.sh` suffices for everything else.
- **Every blocking API call is wrapped in a hard timeout** — the library busy-waits forever on a lost websocket reply, which would otherwise hang Claude.

---

## Status

Two-month broker history and a decade of external spot data both show a consistent, calibrated signal (independent-trade win rates ~56–62% vs. a ~53.5% break-even), with a meta-filter lifting the best decile higher. **None of it counts until the live forward test passes** — that, and a subsequent demo-balance execution trial to measure real fill/spread costs, are the only things that separate an in-sample story from a validated edge. Follow the honest record in [`FORWARD_TEST.md`](FORWARD_TEST.md).
