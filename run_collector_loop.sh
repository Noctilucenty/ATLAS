#!/bin/zsh
# Stopgap collector: runs the normal hourly cycle in a foreground loop.
#
# Why this exists: the launchd agent cannot read this project after a reboot
# because ~/Desktop is TCC-protected and the Full Disk Access grant does not
# survive for launchd-spawned processes (agents exit 78 with no output). A
# terminal you launched yourself already has file access, so a loop started
# from there works where launchd does not.
#
# Usage:  caffeinate -i ./run_collector_loop.sh
# (caffeinate keeps the Mac awake so collection is not interrupted by sleep.)
# Stop with Ctrl-C. This is a stopgap - moving the project off ~/Desktop is
# the durable fix.
cd "$(dirname "$0")"
mkdir -p logs
echo "collector loop started $(date -u +%Y-%m-%dT%H:%M:%SZ) - Ctrl-C to stop"
while true; do
  ./collect_cycle.sh
  echo "cycle done $(date -u +%H:%M:%SZ) exit=$? - next in 1h"
  sleep 3600
done
