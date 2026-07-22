#!/bin/zsh
# Stopgap paper-signal runner: same reason as run_collector_loop.sh - the
# launchd agent cannot read this project after a reboot (~/Desktop is
# TCC-protected), so it exits 78 with no output. A terminal you launched
# yourself has the file access launchd lacks.
#
# Usage:  caffeinate -i ./run_paper_loop.sh
# Stop with Ctrl-C.
#
# Each pass runs 57 minutes then reconnects: the vendored iqoptionapi
# websocket degrades over long sessions (get_candles timeouts), and a fresh
# login each hour clears it.
cd "$(dirname "$0")"
mkdir -p logs
echo "paper loop started $(date -u +%Y-%m-%dT%H:%M:%SZ) - Ctrl-C to stop"
while true; do
  .venv/bin/python live_h2_runner.py --minutes 57
  echo "pass done $(date -u +%H:%M:%SZ) - reconnecting"
  sleep 5
done
