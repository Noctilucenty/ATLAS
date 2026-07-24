"""The per-order PRACTICE guard must be present and structurally correct.

A behavioural test would need the whole broker session, so this pins the
guard at the source level: the buy call must be reachable only underneath a
balance-mode check, and the skip must be recorded rather than swallowed.
The failure this prevents (a REAL-balance order) is unrecoverable, so the
guard's presence is worth asserting even coarsely.
"""

import ast
from pathlib import Path

RUNNER = Path(__file__).resolve().parent.parent / "live_h2_runner.py"


def _buy_call_nodes(tree):
    """Every `_call(client.buy, ...)` node in the runner."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "_call"):
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Attribute) and first.attr == "buy":
            out.append(node)
    return out


def test_runner_has_exactly_one_buy_path():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    assert len(_buy_call_nodes(tree)) == 1, (
        "more than one buy path means the PRACTICE guard can be bypassed")


def test_every_buy_is_guarded_by_a_balance_mode_check():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    buy = _buy_call_nodes(tree)[0]

    # Walk the enclosing statements: some ancestor `if` must test a value
    # derived from get_balance_mode against "PRACTICE".
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "PRACTICE" not in test_src and "mode" not in test_src:
            continue
        if any(child is buy for child in ast.walk(node)):
            guarded = True
            break
    assert guarded, "the buy call is not underneath a balance-mode check"
    assert "get_balance_mode" in source
    assert "balance_mode_" in source, "a refused order must be recorded"


def test_startup_practice_assertion_still_present():
    # The per-order guard supplements the startup check; it does not replace
    # it (a session that starts on REAL must never reach the loop).
    source = RUNNER.read_text(encoding="utf-8")
    assert 'refusing to run: balance mode is not PRACTICE' in source
