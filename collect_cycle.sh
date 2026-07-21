#!/bin/zsh
# One collection cycle: 2h of 1m candles for spot EURUSD and the separate
# EURUSD-OTC market (overlap is deduplicated by the canonical history merge),
# plus a prospective payout snapshot. Payout keys: spot binaries quote under
# EURUSD-op; OTC under EURUSD-OTC (see collector.payout_candidates).
# Scheduled hourly by ~/Library/LaunchAgents/com.atlas.iqoption-collector.plist
cd "$(dirname "$0")"
mkdir -p logs
{
  echo "=== cycle $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python collector.py candles EURUSD EURUSD-OTC --interval 60 --hours 2
  .venv/bin/python collector.py payouts
} >> logs/collector.log 2>&1
