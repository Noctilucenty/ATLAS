#!/bin/zsh
# One-glance ATLAS status: agents, data freshness, paper signals, research runs.
cd "$(dirname "$0")"
SCRATCH="$(dirname "$0")/research_logs"

echo "=== launchd agents (PID / last exit / label) ==="
launchctl list | grep atlas || echo "  none loaded"

echo "\n=== data freshness ==="
.venv/bin/python - << 'EOF'
import duckdb, time
try:
    con = duckdb.connect("market.duckdb", read_only=True)
    n, latest = con.sql("SELECT count(*), max(c.from_ts) FROM candles c").fetchone()
    ns, snap = con.sql("SELECT count(*), max(ts_epoch) FROM payout_snapshots").fetchone()
    age = (time.time() - latest) / 60
    print(f"  candles: {n:,} (latest {age:.0f} min ago)")
    print(f"  payout snapshots: {ns} (latest {(time.time()-snap)/60:.0f} min ago)")
except Exception as exc:
    print(f"  db busy or unreadable: {exc}")
EOF

echo "\n=== paper signals (logs/live_h2.jsonl) ==="
if [[ -f logs/live_h2.jsonl ]]; then
  echo "  total: $(wc -l < logs/live_h2.jsonl | tr -d ' ')"
  tail -3 logs/live_h2.jsonl | sed 's/^/  /'
else
  echo "  none yet (expected ~6/day at the 0.03 gate)"
fi

echo "\n=== research runs (scratchpad logs, newest first) ==="
ls -t $SCRATCH/*.log 2>/dev/null | head -6 | while read f; do
  echo "  --- $(basename $f): $(tail -1 $f | cut -c1-100)"
done

echo "\n=== last collector cycle ==="
grep "cycle exit status" logs/collector.log | tail -2 | sed 's/^/  /'
