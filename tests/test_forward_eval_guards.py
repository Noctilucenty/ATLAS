"""Verdict-day guards. forward_eval runs exactly ONCE, so a crash in it is
unrecoverable - these pin failure modes an audit actually found.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_load_bundle_warning_branch_does_not_raise_nameerror():
    """meta-h3.pkl carries no data_end_ts, so load_bundle's 'cannot verify'
    warning branch executes on EVERY run. It referenced sys.stderr while the
    module never imported sys, raising NameError - and main() caught only
    SystemExit, so it would have killed the leak-immune paper track too.
    """
    pytest.importorskip("lightgbm")
    from forward_eval import load_bundle

    if not (PROJECT_DIR / "models" / "meta-h3.pkl").exists():
        pytest.skip("models/meta-h3.pkl not present")
    name, bundle = load_bundle("meta-h3.pkl", 1784678400)
    assert name == "meta-h3.pkl"
    assert isinstance(bundle, dict)


def test_forward_eval_imports_sys_for_its_stderr_writes():
    source = (PROJECT_DIR / "forward_eval.py").read_text(encoding="utf-8")
    if "sys.stderr" in source:
        assert "import sys" in source, (
            "forward_eval writes to sys.stderr but does not import sys")


def test_candles_track_failure_cannot_kill_the_paper_track():
    """main() must contain ANY candles-track exception, not just SystemExit:
    the paper track is independent evidence and immune to hindsight."""
    source = (PROJECT_DIR / "forward_eval.py").read_text(encoding="utf-8")
    assert "except (SystemExit, Exception)" in source, (
        "a candles-track failure is not broadly contained")


@pytest.mark.parametrize("module", [
    "forward_eval.py", "live_h2_runner.py", "mission_control.py",
    "settle_missing.py", "probe_execution.py", "probe_order_record.py",
    "selfcheck.py", "watchdog.py", "catchup_gaps.py", "extra_collect.py",
    "backup_state.py", "status.py", "dashboard.py",
])
def test_no_undefined_names(module):
    """Catch the whole CLASS of the sys-import bug: pylint's undefined-variable
    check over every operational module. A NameError on a rarely-taken branch
    is exactly the kind of defect that only shows up when it matters."""
    out = subprocess.run(
        [sys.executable, "-m", "pylint", "--disable=all",
         "--enable=E0602,E0601,E1101", "--score=n", module],
        cwd=PROJECT_DIR, capture_output=True, text=True, timeout=300)
    # E1101 (no-member) can false-positive on C extensions; only fail on the
    # undefined-name families that are unambiguous.
    hard = [ln for ln in out.stdout.splitlines()
            if ": E0602" in ln or ": E0601" in ln]
    assert not hard, "undefined names:\n" + "\n".join(hard)
