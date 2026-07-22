"""Operational regression test for collect_cycle.sh.

Runs the ACTUAL zsh script (the one launchd executes) in a sandbox with a
stub .venv/bin/python, because the Python suite alone cannot catch shell
failures - e.g. `status=0` dying instantly on zsh's read-only special
parameter, which silently broke hourly collection in production."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_DIR / "collect_cycle.sh"

pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not available")


def stage_sandbox(tmp_path: Path) -> Path:
    """Copy the real script next to a stub python that exits per $STUB_EXIT."""
    script = tmp_path / "collect_cycle.sh"
    shutil.copy(SCRIPT, script)
    stub = tmp_path / ".venv" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    # The stub must emulate python well enough for the script's real usage:
    # collect_cycle.sh derives its asset list by importing the registry.
    stub.write_text(
        "#!/bin/zsh\n"
        'if [[ "$*" == *"from instruments import"* ]]; then\n'
        '  echo "EURUSD EURUSD-OTC"\n'
        "  exit 0\n"
        "fi\n"
        'echo "STUB python $@"\n'
        "exit ${STUB_EXIT:-0}\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return script


def run_cycle(script: Path, stub_exit: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("STUB_EXIT", None)
    if stub_exit is not None:
        env["STUB_EXIT"] = stub_exit
    return subprocess.run(
        ["/bin/zsh", str(script)], env=env, capture_output=True, text=True, timeout=30
    )


def test_script_completes_and_logs_a_full_cycle(tmp_path):
    script = stage_sandbox(tmp_path)
    result = run_cycle(script)
    log = (tmp_path / "logs" / "collector.log").read_text()
    # A zsh-level crash (like the read-only `status` bug) fails all of these:
    assert result.returncode == 0, result.stderr
    assert "=== cycle " in log
    assert "STUB python collector.py candles EURUSD EURUSD-OTC" in log
    assert "STUB python collector.py payouts" in log
    assert "STUB python health_report.py --current-cycle-status 0" in log
    assert "=== cycle exit status: 0 ===" in log


def test_collector_failure_propagates_to_launchd(tmp_path):
    script = stage_sandbox(tmp_path)
    result = run_cycle(script, stub_exit="2")
    log = (tmp_path / "logs" / "collector.log").read_text()
    assert result.returncode != 0
    assert "=== cycle exit status: 1 ===" in log
    # The health report must see THIS cycle's failure, not last cycle's.
    assert "STUB python health_report.py --current-cycle-status 1" in log


def test_script_never_assigns_zsh_reserved_status_variable():
    body = SCRIPT.read_text()
    for line in body.splitlines():
        code = line.split("#")[0]
        assert "status=" not in code.replace("cycle_status=", ""), (
            f"reserved zsh variable assignment in: {line!r}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
