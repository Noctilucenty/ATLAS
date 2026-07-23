# ATLAS auto-catchup: keeps the forward-test dataset current without any
# always-on process. Opt in by adding this line to ~/.zshrc:
#
#   [ -f ~/dev/ATLAS/ATLAS/atlas_hook.zsh ] && source ~/dev/ATLAS/ATLAS/atlas_hook.zsh
#
# Why a shell hook: launchd cannot read this project after a reboot because
# ~/Desktop is TCC-protected (agents exit 78), but a terminal you launched
# yourself already has file access. Opening a terminal is therefore the one
# reliable trigger available, and you open terminals anyway.
#
# Behaviour: on shell start, if the data is stale and no catchup has run in
# the last THROTTLE seconds, run one in the background. Never blocks the
# prompt, never runs twice concurrently, and stays silent unless it starts.

ATLAS_DIR="${ATLAS_DIR:-$HOME/dev/ATLAS/ATLAS}"
ATLAS_THROTTLE=${ATLAS_THROTTLE:-21600}   # 6h between automatic catchups

atlas_catchup() {
  ( cd "$ATLAS_DIR" 2>/dev/null && ./catchup.sh "$@" ) || return 1
}

atlas_status() {
  ( cd "$ATLAS_DIR" 2>/dev/null && ./status.sh ) || return 1
}

_atlas_auto_catchup() {
  [ -d "$ATLAS_DIR" ] || return 0
  local marker="$ATLAS_DIR/.last_catchup" lock="$ATLAS_DIR/.catchup_lock"
  local now=$(date +%s) last=0

  # A stale lock from a killed run must not disable catchup forever.
  if [ -f "$lock" ]; then
    local lock_age=$(( now - $(cat "$lock" 2>/dev/null || echo 0) ))
    [ "$lock_age" -lt 3600 ] && return 0
  fi
  [ -f "$marker" ] && last=$(cat "$marker" 2>/dev/null || echo 0)
  [ $(( now - last )) -lt "$ATLAS_THROTTLE" ] && return 0

  echo "[atlas] data is stale - catching up in the background (atlas_status to check)"
  (
    echo "$now" > "$lock"
    cd "$ATLAS_DIR" && ./catchup.sh >> logs/auto_catchup.log 2>&1
    rm -f "$lock"
  ) &!
}

# Run once per interactive shell, never in scripts or non-interactive use.
[[ -o interactive ]] && _atlas_auto_catchup

# --- agent health watch (added 2026-07-24) ---------------------------------
# launchd re-fires missed calendar jobs on wake, so the agents self-resume
# after sleep; this is the belt-and-suspenders layer for the cases where a
# wake-time start failed, plus a visible macOS notification so a silent
# outage cannot go unnoticed. Runs on interactive shell start, like catchup.

_atlas_agent_watch() {
  [ -d "$ATLAS_DIR" ] || return 0
  local hb="$ATLAS_DIR/logs/live_h2_heartbeat.jsonl" now=$(date +%s)
  local notify=""
  # Paper/trade agent: heartbeat older than 10 min while loaded = stale.
  if launchctl list 2>/dev/null | grep -q com.atlas.h2-paper; then
    local hb_age=999999
    [ -f "$hb" ] && hb_age=$(( now - $(stat -f %m "$hb" 2>/dev/null || echo 0) ))
    if [ "$hb_age" -gt 4500 ]; then
      launchctl kickstart "gui/$(id -u)/com.atlas.h2-paper" 2>/dev/null
      notify="trade agent restarted (heartbeat was $((hb_age/60))m old)"
    fi
  fi
  # Collector: log untouched for > 2h = missed cycles; kickstart one now.
  local cl="$ATLAS_DIR/logs/collector.log"
  if [ -f "$cl" ] && [ $(( now - $(stat -f %m "$cl") )) -gt 7200 ]; then
    launchctl kickstart "gui/$(id -u)/com.atlas.iqoption-collector" 2>/dev/null
    notify="${notify:+$notify; }collector kicked after gap"
  fi
  if [ -n "$notify" ]; then
    osascript -e "display notification \"$notify\" with title \"ATLAS\"" 2>/dev/null
    echo "[atlas] $notify"
  fi
}

[[ -o interactive ]] && _atlas_agent_watch
