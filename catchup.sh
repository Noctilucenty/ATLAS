#!/bin/zsh
# Backfill exactly what this Mac missed while it was off, then snapshot payouts.
#
# IQ Option serves ~60 days of 1-minute history on demand, so candles are
# retroactively recoverable and continuous uptime is unnecessary. This reads
# the newest stored candle and fetches only the hours since (plus a small
# overlap; the canonical history merge deduplicates). Only payout snapshots
# are point-in-time and unrecoverable, so one is taken per run.
#
# Usage:  ./catchup.sh [hours]   - hours overrides the auto-detected gap
cd "$(dirname "$0")"
mkdir -p logs research_logs

# Auto-detect the gap. Falls back to 168h if the DB is locked or missing.
HOURS=${1:-$(.venv/bin/python - <<'PY' 2>/dev/null || echo 168
import duckdb, time, math
try:
    con = duckdb.connect("market.duckdb", read_only=True)
    latest = con.sql("SELECT max(from_ts) FROM candles").fetchone()[0]
    gap_h = (time.time() - latest) / 3600
    # +2h overlap covers the partially-collected boundary hour; the broker
    # only retains ~60 days, so asking for more than that is wasted work.
    print(min(max(math.ceil(gap_h) + 2, 2), 1400))
except Exception:
    print(168)
PY
)}

# Registry is the single source of truth for what we collect.
ASSETS=(${=$(.venv/bin/python -c "from instruments import INSTRUMENTS; print(' '.join(INSTRUMENTS))")})

echo "[$(date -u +%H:%MZ)] backfilling ${HOURS}h for ${#ASSETS} instruments..."
CATCHUP_OK=1
.venv/bin/python collector.py candles $ASSETS --interval 60 --hours $HOURS \
  >> logs/catchup.log 2>&1 || { echo "WARN: some candle fetches failed (logs/catchup.log)"; CATCHUP_OK=0; }
.venv/bin/python collector.py payouts >> logs/catchup.log 2>&1 \
  || echo "WARN: payout snapshot failed"

.venv/bin/python - <<'PY'
import duckdb, time
con = duckdb.connect("market.duckdb", read_only=True)
n, latest = con.sql("SELECT count(*), max(from_ts) FROM candles").fetchone()
snaps = con.sql("SELECT count(*) FROM payout_snapshots").fetchone()[0]
print(f"candles {n:,} (newest {(time.time()-latest)/60:.0f} min old), "
      f"payout snapshots {snaps}")
PY
# Only stamp on success - a total failure must not throttle retries for 6h.
[ "$CATCHUP_OK" = "1" ] && date +%s > .last_catchup
