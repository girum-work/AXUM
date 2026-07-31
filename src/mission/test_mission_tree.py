"""
AXUM ROVER - Fault-injection tests for the behavior tree engine.

WHY these matter more than normal unit tests: a mission tree's entire
purpose is handling things going wrong. A test suite that only exercises
the happy path proves nothing about the actual reason this architecture
was chosen over a linear script. Every test here deliberately breaks
something and checks the tree recovers (or correctly gives up) the way
it's supposed to.

These tests only depend on behavior_tree.py — no controller.py, no
Arduino, no repo-specific analysis modules — so they run anywhere,
including a laptop with nothing plugged in.

Run with: pytest test_mission_tree.py -v
"""

from __future__ import annotations

from behavior_tree import (
    ActionNode,
    Blackboard,
    ConditionNode,
    ConfidenceGate,
    FaultInjector,
    Fallback,
    Invariant,
    Parallel,
    Retry,
    Sequence,
    Status,
    Timeout,
    run_to_completion,
)


def _always_succeed(bb: Blackboard) -> Status:
    return Status.SUCCESS


def test_sequence_all_success() -> None:
    tree = Sequence("S", [ActionNode("A", _always_succeed), ActionNode("B", _always_succeed)])
    assert run_to_completion(tree, Blackboard()) == Status.SUCCESS


def test_sequence_short_circuits_on_failure() -> None:
    calls = []

    def a(bb):
        calls.append("A")
        return Status.FAILURE

    def b(bb):
        calls.append("B")
        return Status.SUCCESS

    tree = Sequence("S", [ActionNode("A", a), ActionNode("B", b)])
    result = run_to_completion(tree, Blackboard())
    assert result == Status.FAILURE
    assert calls == ["A"]  # B should never have been ticked


def test_retry_recovers_after_transient_failures() -> None:
    """A node that fails twice then succeeds should end in SUCCESS with Retry(max_attempts=3)."""
    state = {"calls": 0}

    def flaky(bb):
        state["calls"] += 1
        if state["calls"] < 3:
            return Status.FAILURE
        return Status.SUCCESS

    tree = Retry("R", ActionNode("Flaky", flaky), max_attempts=3)
    result = run_to_completion(tree, Blackboard())
    assert result == Status.SUCCESS
    assert state["calls"] == 3


def test_retry_gives_up_after_max_attempts() -> None:
    tree = Retry("R", ActionNode("AlwaysFails", lambda bb: Status.FAILURE), max_attempts=3)
    result = run_to_completion(tree, Blackboard())
    assert result == Status.FAILURE


def test_fault_injector_forces_failure_on_named_node() -> None:
    """The core promise of fault injection: force ONE named node to fail without touching the action function."""
    bb = Blackboard()
    injector = FaultInjector()
    injector.force_failure("Pick")
    bb.set("fault_injector", injector)

    tree = ActionNode("Pick", _always_succeed)  # would succeed if not intercepted
    result = run_to_completion(tree, bb)
    assert result == Status.FAILURE


def test_fault_injector_plus_retry_recovers() -> None:
    """
    Simulates: PICK fails twice due to an injected fault, then the fault
    is cleared (mimicking a transient real-world failure resolving), and
    the retry succeeds on the next attempt. This is the exact scenario a
    stuck vacuum-seal grip is supposed to trigger and recover from.

    NOTE: FaultInjector intercepts BEFORE the wrapped action runs, so the
    action function itself never executes while a fault is forced — the
    fault has to be cleared externally (by the test, standing in for
    "the real-world glitch went away"), not by the action noticing it's
    been called before.
    """
    bb = Blackboard()
    injector = FaultInjector()
    injector.force_failure("Pick")
    bb.set("fault_injector", injector)

    tree = Retry("PickRetry", ActionNode("Pick", _always_succeed), max_attempts=5)

    status = tree.tick(bb)
    assert status == Status.RUNNING  # forced failure #1, Retry says "try again"
    status = tree.tick(bb)
    assert status == Status.RUNNING  # forced failure #2, still retrying

    injector.clear()  # the transient fault resolves

    status = tree.tick(bb)
    assert status == Status.SUCCESS  # action finally allowed to actually run


def test_confidence_gate_hard_aborts_below_threshold() -> None:
    def report_low_confidence(bb):
        bb.set("grip_confidence", 0.05)
        return Status.SUCCESS

    gated = ConfidenceGate("Gate", ActionNode("Pick", report_low_confidence), "grip_confidence", retry_below=0.5, abort_below=0.15)
    bb = Blackboard()
    result = run_to_completion(gated, bb)
    assert result == Status.FAILURE
    assert bb.get("grip_confidence_hard_abort") is True


def test_confidence_gate_allows_retry_band_to_recover() -> None:
    """
    Confidence starts marginal (retriable, not hard-abort), then improves
    on a later attempt — Retry wrapping the gate should let it through.
    """
    state = {"calls": 0}

    def report_confidence(bb):
        state["calls"] += 1
        bb.set("grip_confidence", 0.4 if state["calls"] == 1 else 0.9)
        return Status.SUCCESS

    gated = ConfidenceGate("Gate", ActionNode("Pick", report_confidence), "grip_confidence", retry_below=0.5, abort_below=0.15)
    tree = Retry("R", gated, max_attempts=3)
    result = run_to_completion(tree, Blackboard())
    assert result == Status.SUCCESS
    assert state["calls"] == 2


def test_invariant_blocks_child_without_ticking_it() -> None:
    """The whole point of Invariant: the guarded action must never even run if the predicate fails."""
    ticked = {"child_ran": False}

    def dangerous_action(bb):
        ticked["child_ran"] = True
        return Status.SUCCESS

    guarded = Invariant("Guard", ActionNode("Transfer", dangerous_action), predicate=lambda bb: False, violation_message="test violation")
    bb = Blackboard()
    result = run_to_completion(guarded, bb)
    assert result == Status.FAILURE
    assert ticked["child_ran"] is False
    assert bb.get("last_invariant_violation") == "test violation"


def test_timeout_fails_a_node_stuck_running() -> None:
    def stuck(bb):
        return Status.RUNNING

    tree = Timeout("T", ActionNode("Stuck", stuck), timeout_seconds=0.05)
    result = run_to_completion(tree, Blackboard(), tick_interval=0.02)
    assert result == Status.FAILURE


def test_supervisor_parallel_interrupts_mission_on_health_failure() -> None:
    """
    This is the core supervisor claim: a battery/E-stop check as a
    Parallel sibling can fail the WHOLE tree even while the mission
    sequence itself is still happily reporting SUCCESS on each phase.
    """
    mission_ticks = {"n": 0}

    def mission_phase(bb):
        mission_ticks["n"] += 1
        return Status.SUCCESS

    def health_check(bb):
        return bb.get("battery_ok", True)

    tree = Parallel("Supervised", [
        ConditionNode("Health", health_check),
        Sequence("Mission", [ActionNode("Phase1", mission_phase)]),
    ])

    bb = Blackboard()
    bb.set("battery_ok", False)  # simulate critical battery from tick 1
    result = run_to_completion(tree, bb)
    assert result == Status.FAILURE


def test_fallback_tries_alternatives_until_one_succeeds() -> None:
    tree = Fallback("F", [
        ActionNode("PrimaryFails", lambda bb: Status.FAILURE),
        ActionNode("BackupSucceeds", lambda bb: Status.SUCCESS),
    ])
    assert run_to_completion(tree, Blackboard()) == Status.SUCCESS


def test_max_ticks_ceiling_prevents_infinite_hang() -> None:
    """A node stuck RUNNING forever with no Timeout wrapper must still resolve, not hang the process."""
    tree = ActionNode("StuckForever", lambda bb: Status.RUNNING)
    result = run_to_completion(tree, Blackboard(), max_ticks=20, tick_interval=0.001)
    assert result == Status.FAILURE