#!/bin/zsh
# Run BOTH the collector and the paper-signal runner together for a bounded
# session, then shut both down cleanly. One paper runner only - a second
# would double-write logs/live_h2.jsonl and corrupt the forward test.
#
# Usage:  caffeinate -i ./run_both.sh [hours]     (default 3)
# Ctrl-C stops both immediately.
cd "$(dirname "$0")"
HOURS=${1:-3}
mkdir -p logs
SECS=$(( HOURS * 3600 ))

# One catch-up first so the dataset is current before the session starts.
./catchup.sh >> logs/run_both.log 2>&1

echo "running collector + paper for ${HOURS}h (until $(date -v+${HOURS}H +%H:%M)) - Ctrl-C to stop"

# Collector: one cycle per hour.
( while true; do ./collect_cycle.sh; sleep 3600; done ) &
COLLECTOR=$!

# Paper: reconnect every 57 min (the vendored websocket degrades over long
# sessions).
( while true; do .venv/bin/python live_h2_runner.py --minutes 57; sleep 5; done ) &
PAPER=$!

# Stop both when the timer expires or on Ctrl-C.
trap 'kill $COLLECTOR $PAPER 2>/dev/null; echo "stopped."; exit 0' INT TERM
sleep $SECS
kill $COLLECTOR $PAPER 2>/dev/null
echo "done after ${HOURS}h - $(wc -l < logs/live_h2.jsonl 2>/dev/null || echo 0) total paper signals"
