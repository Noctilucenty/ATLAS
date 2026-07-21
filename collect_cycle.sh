#!/bin/zsh
# One collection cycle: 2h of 1m EURUSD candles (overlap is deduplicated by
# the canonical history merge) plus a prospective payout snapshot.
# Scheduled hourly by ~/Library/LaunchAgents/com.atlas.iqoption-collector.plist
cd "$(dirname "$0")"
mkdir -p logs
{
  echo "=== cycle $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python collector.py candles EURUSD --interval 60 --hours 2
  .venv/bin/python collector.py payouts
} >> logs/collector.log 2>&1
