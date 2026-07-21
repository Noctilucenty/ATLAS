#!/bin/zsh
# One collection cycle: 2h of 1m candles for spot EURUSD and the separate
# EURUSD-OTC market (overlap is deduplicated by the canonical history merge),
# plus a prospective payout snapshot. Broker key mapping lives in
# instruments.py (spot binaries quote under EURUSD-op; OTC under EURUSD-OTC).
# Scheduled hourly by ~/Library/LaunchAgents/com.atlas.iqoption-collector.plist
# Exits nonzero if candle collection stored nothing or the payout snapshot
# failed, so launchd's LastExitStatus reflects real collection health.
cd "$(dirname "$0")"
mkdir -p logs
status=0
{
  echo "=== cycle $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python collector.py candles EURUSD EURUSD-OTC --interval 60 --hours 2 || status=1
  .venv/bin/python collector.py payouts || status=1
  echo "=== cycle exit status: $status ==="
} >> logs/collector.log 2>&1
exit $status
