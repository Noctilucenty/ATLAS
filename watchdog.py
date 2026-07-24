"""ATLAS watchdog - runs every 15 min from Task Scheduler (interactive,
pythonw.exe so no console window). Fast read-only health check; Windows
toast on CRITICAL, re-alerting at most every 2 h while the condition
persists, plus one recovery toast when health returns.

State in logs/watchdog_state.json, log in logs/watchdog.log. Never touches
trading state.
"""

import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mission_control import LOGS, PROJECT_DIR, build_status

STATE_PATH = LOGS / "watchdog_state.json"
LOG_PATH = LOGS / "watchdog.log"
REALERT_S = 2 * 3600
DASHBOARD_PORT = 8787


def log(msg: str) -> None:
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {msg}\n")


def toast(title: str, body: str) -> bool:
    """Windows toast via WinRT (no extra packages). Falls back silently -
    the watchdog log always has the event either way."""
    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$xml = @'
<toast scenario="urgent"><visual><binding template="ToastGeneric">
<text>{title}</text><text>{body}</text>
</binding></visual></toast>
'@
$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xml)
$n = [Windows.UI.Notifications.ToastNotification]::new($doc)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('ATLAS').Show($n)
"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def ensure_dashboard() -> None:
    """Self-heal the dashboard: if nothing answers on its port, spawn it
    detached with pythonw (no window). At most one spawn per watchdog run;
    dashboard.py binding the port is the only success criterion, so a crash
    just means another try in 15 minutes."""
    if port_listening(DASHBOARD_PORT):
        return
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | \
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        [str(exe), str(PROJECT_DIR / "dashboard.py")],
        cwd=PROJECT_DIR, creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    log(f"dashboard not listening on {DASHBOARD_PORT} - respawned")


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def main() -> int:
    now = int(time.time())
    status = build_status(now=now, deep=False)
    tier, reasons = status["tier"], status["reasons"]
    state = load_state()
    prev_tier = state.get("tier", "HEALTHY")
    last_alert = state.get("last_alert_ts", 0)

    log(f"tier={tier}" + (f" reasons={'; '.join(reasons)}" if reasons else ""))

    if tier == "CRITICAL":
        if prev_tier != "CRITICAL" or now - last_alert >= REALERT_S:
            ok = toast("ATLAS CRITICAL - trader down",
                       "; ".join(reasons)[:180] or "unknown")
            log(f"alert sent (toast={'ok' if ok else 'failed'})")
            state["last_alert_ts"] = now
    elif prev_tier == "CRITICAL":
        toast("ATLAS recovered", "health back to " + tier)
        log("recovery alert sent")

    ensure_dashboard()

    state["tier"] = tier
    LOGS.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
