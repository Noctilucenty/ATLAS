"""Supervisor single-instance lock.

2026-07-29 audit: Task Scheduler had lost track of a running supervisor - a
`schtasks /run` rejected by IgnoreNew left the task 'Ready' while the real
process kept going. Any later recovery, including watchdog.py's automated
restart, would then have started a SECOND supervisor: duplicate collection,
duplicate runner spawning, and churn against the runner's own lock. The runner
has had this protection for a while; the supervisor had none.
"""

import supervisor

PORT = 47391  # test-only, away from 47201 (runner) and 47202 (supervisor)


def test_second_acquire_is_refused():
    first = supervisor.acquire_singleton_lock(PORT)
    assert first is not None
    try:
        assert supervisor.acquire_singleton_lock(PORT) is None
    finally:
        first.close()


def test_lock_is_released_when_the_holder_closes():
    """The OS reclaims the port on exit or crash, so there is no stale
    lockfile to clean up after an ungraceful death."""
    first = supervisor.acquire_singleton_lock(PORT + 1)
    assert first is not None
    first.close()
    second = supervisor.acquire_singleton_lock(PORT + 1)
    assert second is not None
    second.close()


def test_supervisor_and_runner_use_different_ports():
    """Sharing a port would make the supervisor and its own child fight."""
    import live_h2_runner

    src = (supervisor.PROJECT_DIR / "live_h2_runner.py").read_text(
        encoding="utf-8")
    assert "47201" in src
    assert supervisor.SUPERVISOR_LOCK_PORT == 47202
    assert live_h2_runner is not None


def test_main_exits_when_the_lock_is_held(monkeypatch):
    """A second supervisor must exit cleanly, not raise and not trade."""
    held = supervisor.acquire_singleton_lock(supervisor.SUPERVISOR_LOCK_PORT)
    if held is None:
        # A real supervisor is running on this machine - that IS the condition
        # under test, so the assertion below still holds.
        pass
    try:
        monkeypatch.setattr("sys.argv", ["supervisor.py", "--paper"])
        started = []
        monkeypatch.setattr(supervisor.subprocess, "Popen",
                            lambda *a, **k: started.append(a) or (_ for _ in ()).throw(
                                AssertionError("second supervisor spawned a runner")))
        supervisor.main()          # must return promptly without spawning
        assert started == []
    finally:
        if held is not None:
            held.close()
