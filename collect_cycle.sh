#!/bin/zsh
# One collection cycle: 2h of 1m candles for spot EURUSD and the separate
# EURUSD-OTC market (overlap is deduplicated by the canonical history merge),
# plus a prospective payout snapshot. Broker key mapping lives in
# instruments.py (spot binaries quote under EURUSD-op; OTC under EURUSD-OTC).
# Scheduled hourly by ~/Library/LaunchAgents/com.atlas.iqoption-collector.plist
# Exits nonzero on ANY failure - partial candle failure, total failure, or a
# payout snapshot missing a required quote key - so launchd's LastExitStatus
# reflects real collection health.
# NOTE: `status` is a READ-ONLY special parameter in zsh; assigning it kills
# the script instantly. Never name a variable `status` here.
cd "$(dirname "$0")"
mkdir -p logs
cycle_status=0
{
  echo "=== cycle $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python collector.py candles EURUSD EURUSD-OTC GBPUSD GBPUSD-OTC USDJPY USDJPY-OTC AUDUSD EURGBP-OTC EURJPY EURJPY-OTC AUDCAD-OTC GBPJPY-OTC NZDUSD-OTC USDCHF-OTC USDSGD-OTC USDZAR-OTC --interval 60 --hours 2 || cycle_status=1
  .venv/bin/python collector.py payouts || cycle_status=1
  # Monitoring only: the health report (logs/health.json) never changes the
  # cycle exit status - LastExitStatus stays a pure collection-health signal.
  # The current status is passed explicitly because this cycle's status line
  # is appended below, AFTER the report runs.
  .venv/bin/python health_report.py --current-cycle-status $cycle_status || true
  echo "=== cycle exit status: $cycle_status ==="
} >> logs/collector.log 2>&1
exit $cycle_status
