"""Supervisor resilience properties, from the 2026-07-28 outage.

Root cause chain: a transient broker problem hung the collector; collect_cycle
ran INLINE with a 1200s subprocess timeout, so the supervisor's main loop was
frozen for twenty minutes; during that freeze a wedged runner could not be
relaunched. A runner hang became a 74-minute trading outage.

Keeping the trader alive outranks collecting candles - a missed collection is
repaired by catchup_gaps within hours, a missed trading minute never is.
"""

import inspect
from pathlib import Path

import supervisor

SRC = Path(__file__).resolve().parent.parent / "supervisor.py"


def _main_loop_source() -> str:
    return inspect.getsource(supervisor.main)


def test_collection_never_blocks_the_supervision_loop():
    body = _main_loop_source()
    assert "threading.Thread" in body, (
        "collect_cycle must run off the main loop; inline collection froze "
        "runner supervision for 20 minutes")
    assert "daemon=True" in body


def test_overlapping_collections_are_skipped_not_queued():
    """A slow collector must not stack threads every cycle."""
    body = _main_loop_source()
    assert "is_alive()" in body


def test_runner_relaunch_has_short_run_backoff():
    """A stray socket-lock holder makes every new runner exit instantly;
    without backoff the loop respawns it every 30s forever."""
    body = _main_loop_source()
    assert "backoff" in body
    assert "SHORT_RUN_S" in body


def test_backoff_is_bounded_and_resets():
    body = _main_loop_source()
    assert "BACKOFF_MAX_S" in body
    # a healthy session must clear the penalty
    assert "backoff = BACKOFF_START_S" in body


def test_children_are_launched_windowless():
    body = _main_loop_source()
    assert "creationflags=NO_WINDOW" in body


def test_supervisor_module_compiles():
    import ast

    ast.parse(SRC.read_text(encoding="utf-8"))
