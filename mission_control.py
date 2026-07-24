"""Mission Control core - shared read-only data layer for status.py,
dashboard.py and watchdog.py on the always-on Windows host.

HARD CONSTRAINTS (see docs/superpowers/specs/2026-07-24-mission-control-design.md):
- Read-only against ALL trading state: market.duckdb is opened read_only,
  journal.db via sqlite URI mode=ro, jsonl logs are only ever read.
- Never evaluates the pre-registered forward-test criteria. Counts and
  displays only; forward_eval.py runs ONCE, by hand, after the window.
- Imports nothing from the signal path (no features/train/analyzer imports).

The candle-label computation here is the LABEL-FIDELITY TRACKER: an
approximation (close of the decision bar vs close of the bar ending nearest
entry+expiry) whose whole purpose is to be compared against the broker's
settled verdict. Disagreement is the measurement, not an error.
"""

import json
import re
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOGS = PROJECT_DIR / "logs"
HEARTBEAT_PATH = LOGS / "live_h2_heartbeat.jsonl"
SIGNALS_PATH = LOGS / "live_h2.jsonl"
SUPERVISOR_LOG = LOGS / "supervisor.log"
MARKET_DB = PROJECT_DIR / "market.duckdb"
JOURNAL_DB = PROJECT_DIR / "journal.db"
TASK_NAME = "ATLAS-supervisor"

EXPIRY_S = 15 * 60          # matches live_h2_runner EXPIRY_MINUTES
HEARTBEAT_STALE_S = 600     # runner cycles each minute; 10 min quiet = down
CANDLE_STALE_S = 9000       # mirrors health_report.STALE_CANDLE_S
CHURN_WINDOW_S = 600        # repeated runner exits inside 10 min = lock churn

# Pre-registered verdict set - DISPLAY ONLY, never evaluated here.
REGISTERED_SET = ("H2p ev0.03", "H2s ev0.04", "H3 meta0.60", "H4")


# ---------------------------------------------------------------- jsonl I/O

def read_jsonl(path: Path, tail: int | None = None) -> list[dict]:
    """Best-effort jsonl reader: skips torn/partial lines (the writer may be
    mid-append when we read)."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if tail is not None:
        lines = lines[-tail:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def split_signals(rows: list[dict]) -> dict:
    """Partition live_h2.jsonl rows. Settled rows are appended duplicates of
    their signal row plus result/profit/settled, so totals must not double
    count them."""
    signals = [r for r in rows if not r.get("settled")]
    settled = [r for r in rows if r.get("settled")]
    placed = [r for r in signals if r.get("order_id")]
    skipped_otc = [r for r in signals if r.get("trade_skipped")]
    return {
        "signals": signals,
        "settled": settled,
        "placed": placed,
        "skipped_otc": skipped_otc,
    }


def expected_value(p_up: float, payout: float) -> float:
    """EV of the better side, mirroring train.decide_action's economics
    without importing the signal path."""
    call_ev = p_up * payout - (1.0 - p_up)
    put_ev = (1.0 - p_up) * payout - p_up
    return max(call_ev, put_ev)


def forward_progress(signals: list[dict]) -> dict:
    """Trade COUNTS toward each pre-registered hypothesis. Display only -
    win rates for the verdict set are forward_eval.py's job, exactly once."""
    counts = dict.fromkeys(REGISTERED_SET, 0)
    for r in signals:
        p, payout = r.get("p_up"), r.get("payout")
        if p is None or payout is None:
            continue
        ev = expected_value(float(p), float(payout))
        if ev > 0.03:
            counts["H2p ev0.03"] += 1
        if ev > 0.04:
            counts["H2s ev0.04"] += 1
        if (r.get("meta_p") or 0.0) >= 0.60 and ev > 0.03:
            counts["H3 meta0.60"] += 1
        if r.get("h4_p") is not None:
            counts["H4"] += 1
    return counts


# ------------------------------------------------------------ process state

def scheduled_task_state(task_name: str = TASK_NAME) -> dict:
    """Task Scheduler view of the supervisor. Uses schtasks (no admin needed
    for /query)."""
    try:
        out = subprocess.run(
            ["schtasks", "/query", "/tn", task_name, "/v", "/fo", "LIST"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:  # schtasks missing = not Windows; report unknown
        return {"exists": None, "error": f"{type(exc).__name__}: {exc}"}
    if out.returncode != 0:
        return {"exists": False, "error": (out.stderr or out.stdout).strip()[:200]}
    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", out.stdout, re.MULTILINE)
        return m.group(1).strip() if m else None
    return {
        "exists": True,
        "status": field("Status"),
        "last_run": field("Last Run Time"),
        "last_result": field("Last Result"),
    }


def supervisor_tail(path: Path = SUPERVISOR_LOG, lines: int = 40) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def parse_supervisor_events(tail_lines: list[str]) -> list[tuple[int, str]]:
    """(epoch, message) for each parseable supervisor log line."""
    events = []
    for line in tail_lines:
        m = re.match(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\]\s+(.*)", line)
        if not m:
            continue
        ts = int(datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
                 .replace(tzinfo=timezone.utc).timestamp())
        events.append((ts, m.group(2)))
    return events


def runner_churn(events: list[tuple[int, str]], now: int,
                 window_s: int = CHURN_WINDOW_S) -> int:
    """Count runner exits inside the recent window - the signature of a
    second runner losing the socket-lock race every 30 s."""
    return sum(1 for ts, msg in events
               if msg.startswith("runner exited") and now - ts <= window_s)


# ---------------------------------------------------------------- databases

def candle_freshness() -> dict:
    """Total candles + latest close-time from market.duckdb, read-only.
    The collector may hold the write lock; 'busy' is a normal answer."""
    if not MARKET_DB.exists():
        return {"exists": False}
    try:
        import duckdb
        conn = duckdb.connect(str(MARKET_DB), read_only=True)
        try:
            n, latest = conn.execute(
                "SELECT count(*), max(to_ts) FROM candles").fetchone()
            ns, snap = conn.execute(
                "SELECT count(*), max(ts_epoch) FROM payout_snapshots").fetchone()
        finally:
            conn.close()
        return {"exists": True, "candles": int(n or 0),
                "latest_to_ts": int(latest) if latest else None,
                "payout_snapshots": int(ns or 0),
                "latest_payout_ts": int(snap) if snap else None}
    except Exception as exc:
        return {"exists": True, "busy": f"{type(exc).__name__}: {exc}"}


def candle_closes(asset: str, ts_list: list[int]) -> dict[int, float]:
    """{to_ts: close} for the requested bar close-times of one asset,
    read-only, joined through datasets like storage.load_canonical_history."""
    if not MARKET_DB.exists() or not ts_list:
        return {}
    try:
        import duckdb
        conn = duckdb.connect(str(MARKET_DB), read_only=True)
        try:
            placeholders = ",".join("?" for _ in ts_list)
            rows = conn.execute(
                f"""SELECT c.to_ts, max(c.close)
                    FROM candles c JOIN datasets d ON c.dataset_id = d.id
                    WHERE d.asset = ? AND d.interval_seconds = 60
                      AND c.to_ts IN ({placeholders})
                    GROUP BY c.to_ts""",
                [asset, *ts_list],
            ).fetchall()
        finally:
            conn.close()
        return {int(t): float(c) for t, c in rows}
    except Exception:
        return {}


def journal_counts() -> dict:
    """Row count from the (MCP-side) sqlite journal, opened read-only."""
    if not JOURNAL_DB.exists():
        return {"exists": False}
    try:
        conn = sqlite3.connect(f"file:{JOURNAL_DB.as_posix()}?mode=ro", uri=True)
        try:
            n = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
        finally:
            conn.close()
        return {"exists": True, "runs": int(n)}
    except Exception as exc:
        return {"exists": True, "busy": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------ label fidelity

def _bar_ts(ts: int) -> int:
    """Close-time of the 1m bar that contains epoch ts."""
    return (ts // 60) * 60 + 60


def label_fidelity(settled: list[dict], closes_fn=candle_closes) -> dict:
    """Broker verdict vs candle-approximation label, per settled REAL order.

    Candle label: entry price = close of the decision bar (bar_to_ts);
    settlement price = close of the bar ending nearest entry+15 min.
    call wins if settle > entry, put wins if settle < entry, tie = equal.
    This is the demo-trial fidelity measurement from CLAUDE.md - agreement
    rate is the result, and low agreement is a FINDING, not a bug here.
    """
    from instruments import INSTRUMENTS  # data map only, not signal logic

    per_asset_ts: dict[str, set[int]] = {}
    for r in settled:
        asset = r.get("asset")
        spec = INSTRUMENTS.get(asset)
        if spec is None or not r.get("order_id"):
            continue
        entry_bar = int(r["bar_to_ts"])
        settle_bar = _bar_ts(int(r["ts"]) + EXPIRY_S)
        per_asset_ts.setdefault(spec.candle_asset, set()).update(
            (entry_bar, settle_bar))

    closes = {a: closes_fn(a, sorted(ts)) for a, ts in per_asset_ts.items()}

    rows, agree, disagree, undetermined = [], 0, 0, 0
    for r in settled:
        asset = r.get("asset")
        spec = INSTRUMENTS.get(asset)
        if spec is None or not r.get("order_id"):
            continue
        entry_bar = int(r["bar_to_ts"])
        settle_bar = _bar_ts(int(r["ts"]) + EXPIRY_S)
        cmap = closes.get(spec.candle_asset, {})
        entry, settle = cmap.get(entry_bar), cmap.get(settle_bar)
        broker = (r.get("result") or "").lower()
        if entry is None or settle is None:
            candle = None
        elif settle == entry:
            candle = "equal"
        else:
            went_up = settle > entry
            is_call = r.get("action") == "binary_call"
            candle = "win" if went_up == is_call else "loose"
        if candle is None:
            undetermined += 1
        elif broker in ("win", "loose", "equal"):
            if broker == candle:
                agree += 1
            else:
                disagree += 1
        rows.append({"ts": r.get("ts"), "asset": asset,
                     "action": r.get("action"), "broker": broker or None,
                     "candle": candle, "profit": r.get("profit")})

    judged = agree + disagree
    return {
        "trades": rows,
        "settled_orders": len(rows),
        "judged": judged,
        "agree": agree,
        "disagree": disagree,
        "undetermined": undetermined,
        "agreement_rate": round(agree / judged, 4) if judged else None,
        "target_trades": 100,
        "note": "candle label is the MID-settlement approximation; "
                "disagreement rate IS the measurement",
    }


# -------------------------------------------------------------- health tier

def classify_health(*, now: int, heartbeat_ts: int | None, task_status: str | None,
                    churn_events: int, latest_candle_ts: int | None,
                    supervisor_seen: bool) -> tuple[str, list[str]]:
    """Three-tier health. CRITICAL = trading is not happening; WARNING =
    trading continues but something needs eyes; HEALTHY otherwise."""
    reasons: list[str] = []
    tier = "HEALTHY"

    def warn(msg):
        nonlocal tier
        reasons.append(msg)
        if tier == "HEALTHY":
            tier = "WARNING"

    def critical(msg):
        nonlocal tier
        reasons.append(msg)
        tier = "CRITICAL"

    if heartbeat_ts is None:
        critical("no heartbeat file - runner has never cycled")
    elif now - heartbeat_ts > HEARTBEAT_STALE_S:
        critical(f"heartbeat stale {now - heartbeat_ts}s (> {HEARTBEAT_STALE_S}s)")

    if task_status is None:
        warn("scheduled task state unreadable")
    elif task_status.lower() != "running":
        critical(f"scheduled task not running (status={task_status})")

    if churn_events >= 3:
        warn(f"{churn_events} runner exits in last {CHURN_WINDOW_S}s - "
             "possible duplicate supervisor (socket-lock churn)")
    if not supervisor_seen:
        warn("supervisor.log missing or unparseable")
    if latest_candle_ts is not None and now - latest_candle_ts > CANDLE_STALE_S:
        warn(f"latest candle {now - latest_candle_ts}s old (> {CANDLE_STALE_S}s)")

    return tier, reasons


# ------------------------------------------------------------- full status

def build_status(now: int | None = None, deep: bool = True) -> dict:
    """Everything status.py / dashboard.py / watchdog.py need, one dict.
    deep=False skips the databases (watchdog fast path)."""
    now = int(now if now is not None else time.time())
    heartbeats = read_jsonl(HEARTBEAT_PATH, tail=500)
    hb_ts = int(heartbeats[-1]["ts"]) if heartbeats else None
    task = scheduled_task_state()
    events = parse_supervisor_events(supervisor_tail())
    churn = runner_churn(events, now)
    rows = read_jsonl(SIGNALS_PATH)
    parts = split_signals(rows)

    candles = candle_freshness() if deep else {}
    latest_candle = candles.get("latest_to_ts")

    tier, reasons = classify_health(
        now=now, heartbeat_ts=hb_ts, task_status=task.get("status"),
        churn_events=churn, latest_candle_ts=latest_candle,
        supervisor_seen=bool(events),
    )

    status = {
        "generated_utc": datetime.fromtimestamp(now, timezone.utc)
                         .isoformat(timespec="seconds"),
        "tier": tier,
        "reasons": reasons,
        "task": task,
        "heartbeat": {
            "last_ts": hb_ts,
            "age_s": (now - hb_ts) if hb_ts else None,
            "last": heartbeats[-1] if heartbeats else None,
        },
        "supervisor": {
            "recent_events": [f"{datetime.fromtimestamp(ts, timezone.utc):%H:%M:%S}Z {msg}"
                              for ts, msg in events[-6:]],
            "runner_churn_recent": churn,
        },
        "signals": {
            "total": len(parts["signals"]),
            "orders_placed": len(parts["placed"]),
            "settled": len(parts["settled"]),
            "otc_skipped": len(parts["skipped_otc"]),
            "last_signal": parts["signals"][-1] if parts["signals"] else None,
        },
        "forward_progress": forward_progress(parts["signals"]),
        "registered_note": "counts only - verdicts belong to forward_eval.py, run once",
    }
    if deep:
        status["market_db"] = candles
        status["journal"] = journal_counts()
        status["fidelity"] = label_fidelity(parts["settled"])
    return status


def tier_exit_code(tier: str) -> int:
    return {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}.get(tier, 2)
