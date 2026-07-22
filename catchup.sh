#!/bin/zsh
# Backfill everything the collector missed while this Mac was off.
#
# IQ Option serves ~60 days of 1-minute history on demand, so candles are
# retroactively recoverable: running this once a week captures every minute,
# and the forward test's candles track needs nothing else. Only payout
# snapshots are point-in-time and unrecoverable - this grabs one per run.
#
# Usage:  ./catchup.sh [hours]     (default 168 = 7 days)
# Safe to over-request: the canonical history merge deduplicates.
cd "$(dirname "$0")"
HOURS=${1:-168}
mkdir -p logs
ASSETS=(EURUSD EURUSD-OTC GBPUSD GBPUSD-OTC USDJPY USDJPY-OTC AUDUSD
        EURGBP-OTC EURJPY EURJPY-OTC AUDCAD-OTC GBPJPY-OTC NZDUSD-OTC
        USDCHF-OTC USDSGD-OTC USDZAR-OTC)
echo "backfilling ${HOURS}h for ${#ASSETS} instruments..."
.venv/bin/python collector.py candles $ASSETS --interval 60 --hours $HOURS \
  >> logs/catchup.log 2>&1 || echo "WARN: some candle fetches failed (see logs/catchup.log)"
.venv/bin/python collector.py payouts >> logs/catchup.log 2>&1 \
  || echo "WARN: payout snapshot failed"
.venv/bin/python -c "
import duckdb, time
con = duckdb.connect('market.duckdb', read_only=True)
n, latest = con.sql('SELECT count(*), max(from_ts) FROM candles').fetchone()
print(f'candles: {n:,}  newest {(time.time()-latest)/60:.0f} min old')
"
