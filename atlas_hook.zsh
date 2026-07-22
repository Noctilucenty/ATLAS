# ATLAS auto-catchup: keeps the forward-test dataset current without any
# always-on process. Opt in by adding this line to ~/.zshrc:
#
#   [ -f ~/Desktop/dev/ATLAS/ATLAS/atlas_hook.zsh ] && source ~/Desktop/dev/ATLAS/ATLAS/atlas_hook.zsh
#
# Why a shell hook: launchd cannot read this project after a reboot because
# ~/Desktop is TCC-protected (agents exit 78), but a terminal you launched
# yourself already has file access. Opening a terminal is therefore the one
# reliable trigger available, and you open terminals anyway.
#
# Behaviour: on shell start, if the data is stale and no catchup has run in
# the last THROTTLE seconds, run one in the background. Never blocks the
# prompt, never runs twice concurrently, and stays silent unless it starts.

ATLAS_DIR="${ATLAS_DIR:-$HOME/Desktop/dev/ATLAS/ATLAS}"
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
