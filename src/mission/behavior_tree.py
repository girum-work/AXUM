"""
AXUM ROVER - Minimal behavior tree engine.

WHY hand-rolled instead of py_trees or another BT library: this is a
two-person team on an 8-week clock. A third-party dependency neither of you
can debug at 1am before competition is a worse bet than ~250 lines of code
you wrote and fully understand. This engine implements exactly what AXUM's
mission tree needs — it is not a general-purpose library and shouldn't grow
into one.

CORE CONCEPTS:
    Status      SUCCESS / FAILURE / RUNNING — every node returns one.
    Blackboard  Shared mutable state all nodes read/write (sensor readings,
                confidence scores, hardware handles, dry_run flag, etc.)
    Composite   Sequence, Fallback, Parallel — combine child nodes.
    Decorator   Retry, Timeout, Invariant, ConfidenceGate, Traced — wrap a
                single child and change how its result is interpreted.
    Leaf        ConditionNode (pure check) / ActionNode (does something).

A tree is re-ticked (root.tick(blackboard)) repeatedly by the caller until
it returns SUCCESS or FAILURE (RUNNING means "still working, call again").
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Blackboard:
    """Shared mutable state every node in the tree can read and write."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly copy for logging/recording (best-effort on odd types)."""
        return {k: v for k, v in self._data.items() if _is_json_safe(v)}


def _is_json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))


# ── base node types ──────────────────────────────────────────────────

class Node:
    name: str = "Node"

    def tick(self, bb: Blackboard) -> Status:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal progress (e.g. retry counters, running index)."""
        pass


class Composite(Node):
    def __init__(self, name: str, children: list[Node]) -> None:
        self.name = name
        self.children = children

    def reset(self) -> None:
        for child in self.children:
            child.reset()


class Sequence(Composite):
    """Ticks children in order. Any FAILURE/RUNNING short-circuits. All SUCCESS -> SUCCESS."""

    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name, children)
        self._index = 0

    def tick(self, bb: Blackboard) -> Status:
        while self._index < len(self.children):
            status = self.children[self._index].tick(bb)
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.FAILURE:
                self._index = 0
                return Status.FAILURE
            self._index += 1
        self._index = 0
        return Status.SUCCESS

    def reset(self) -> None:
        super().reset()
        self._index = 0


class Fallback(Composite):
    """AKA Selector. Ticks children until one SUCCEEDS. All FAILURE -> FAILURE."""

    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name, children)
        self._index = 0

    def tick(self, bb: Blackboard) -> Status:
        while self._index < len(self.children):
            status = self.children[self._index].tick(bb)
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.SUCCESS:
                self._index = 0
                return Status.SUCCESS
            self._index += 1
        self._index = 0
        return Status.FAILURE

    def reset(self) -> None:
        super().reset()
        self._index = 0


class Parallel(Composite):
    """
    Ticks ALL children every call, no short-circuiting.

    This is what makes the supervisor pattern work: a battery/E-stop
    monitor and the actual mission sequence are siblings under one
    Parallel. If the monitor reports FAILURE, the whole Parallel reports
    FAILURE immediately — regardless of what phase the mission sequence
    is mid-way through. That's a global interrupt without adding a
    battery check to all 8 mission phases individually.
    """

    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name, children)

    def tick(self, bb: Blackboard) -> Status:
        statuses = [child.tick(bb) for child in self.children]
        if Status.FAILURE in statuses:
            return Status.FAILURE
        if Status.RUNNING in statuses:
            return Status.RUNNING
        return Status.SUCCESS


class Decorator(Node):
    def __init__(self, name: str, child: Node) -> None:
        self.name = name
        self.child = child

    def reset(self) -> None:
        self.child.reset()


class Retry(Decorator):
    """
    Retries the child on FAILURE, up to max_attempts, before giving up
    for real. Reports RUNNING (not FAILURE) between attempts so the
    caller's tick loop keeps calling — a retry is "still working," not
    "done and failed."
    """

    def __init__(self, name: str, child: Node, max_attempts: int = 3) -> None:
        super().__init__(name, child)
        self.max_attempts = max_attempts
        self._attempts = 0

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        if status == Status.FAILURE:
            self._attempts += 1
            if self._attempts < self.max_attempts:
                logger.debug(f"{self.name}: attempt {self._attempts} failed, retrying")
                self.child.reset()
                return Status.RUNNING
            logger.warning(f"{self.name}: exhausted {self.max_attempts} attempts, giving up")
            self._attempts = 0
            return Status.FAILURE
        if status == Status.SUCCESS:
            self._attempts = 0
        return status

    def reset(self) -> None:
        super().reset()
        self._attempts = 0


class Timeout(Decorator):
    """Fails the child if it stays RUNNING longer than timeout_seconds."""

    def __init__(self, name: str, child: Node, timeout_seconds: float) -> None:
        super().__init__(name, child)
        self.timeout_seconds = timeout_seconds
        self._start: float | None = None

    def tick(self, bb: Blackboard) -> Status:
        if self._start is None:
            self._start = time.monotonic()
        if time.monotonic() - self._start > self.timeout_seconds:
            logger.warning(f"{self.name}: timed out after {self.timeout_seconds}s")
            self.child.reset()
            self._start = None
            return Status.FAILURE
        status = self.child.tick(bb)
        if status != Status.RUNNING:
            self._start = None
        return status

    def reset(self) -> None:
        super().reset()
        self._start = None


class Invariant(Decorator):
    """
    Formal safety-invariant guard. Checks `predicate(blackboard)` BEFORE
    the child is allowed to tick at all. If the guard fails, the child
    never runs — this returns FAILURE immediately.

    Example: never enter the transfer-to-turntable action unless the
    blackboard says the vacuum gripper actually confirmed a seal first.
    """

    def __init__(
        self,
        name: str,
        child: Node,
        predicate: Callable[[Blackboard], bool],
        violation_message: str = "",
    ) -> None:
        super().__init__(name, child)
        self.predicate = predicate
        self.violation_message = violation_message or f"Invariant violated: {name}"

    def tick(self, bb: Blackboard) -> Status:
        if not self.predicate(bb):
            logger.error(self.violation_message)
            bb.set("last_invariant_violation", self.violation_message)
            return Status.FAILURE
        return self.child.tick(bb)


class ConfidenceGate(Decorator):
    """
    Reinterprets a child's SUCCESS against a confidence score the child
    wrote to the blackboard, instead of trusting a raw boolean result.

    Below abort_below: treat as a hard FAILURE — the situation won't
    improve with a retry (e.g. ArUco marker not visible at all).
    Between abort_below and retry_below: FAILURE, but an enclosing Retry
    decorator is expected to try again (e.g. marker visible but pose
    estimate is shaky — worth one more attempt).
    At/above retry_below: SUCCESS passes through unchanged.
    """

    def __init__(
        self,
        name: str,
        child: Node,
        confidence_key: str,
        retry_below: float = 0.6,
        abort_below: float = 0.25,
    ) -> None:
        super().__init__(name, child)
        self.confidence_key = confidence_key
        self.retry_below = retry_below
        self.abort_below = abort_below

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        if status != Status.SUCCESS:
            return status
        confidence = bb.get(self.confidence_key)
        if confidence is None:
            # Child didn't report a confidence score — trust the raw status
            # rather than fail on a missing key.
            return status
        if confidence < self.abort_below:
            bb.set("last_abort_reason", f"{self.confidence_key}={confidence:.2f} below abort threshold")
            bb.set(f"{self.confidence_key}_hard_abort", True)
            return Status.FAILURE
        if confidence < self.retry_below:
            return Status.FAILURE
        return Status.SUCCESS


class Traced(Decorator):
    """Wraps a node so every tick is logged to a MissionRecorder."""

    def __init__(self, name: str, child: Node, recorder: "MissionRecorder") -> None:
        super().__init__(name, child)
        self.recorder = recorder

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        self.recorder.record(self.name, status, bb)
        return status


# ── leaves ────────────────────────────────────────────────────────────

class ConditionNode(Node):
    """Pure check against the blackboard. No side effects, no hardware calls."""

    def __init__(self, name: str, predicate: Callable[[Blackboard], bool]) -> None:
        self.name = name
        self.predicate = predicate

    def tick(self, bb: Blackboard) -> Status:
        return Status.SUCCESS if self.predicate(bb) else Status.FAILURE


class ActionNode(Node):
    """
    Leaf that runs a real action. `action(bb)` should return a Status —
    SUCCESS/FAILURE for actions that block until done (which is how
    ArmController/TurntableController/DriveController already behave),
    or RUNNING if you want a long action to be polled across ticks.

    Fault injection is transparent here: if the blackboard has a
    FaultInjector under "fault_injector", it's checked before the real
    action runs, so tests can force this specific node to fail/hang
    without the action function itself knowing it's being tested.
    """

    def __init__(self, name: str, action: Callable[[Blackboard], Status]) -> None:
        self.name = name
        self.action = action

    def tick(self, bb: Blackboard) -> Status:
        injector: FaultInjector | None = bb.get("fault_injector")
        if injector is not None:
            forced = injector.check(self.name)
            if forced is not None:
                logger.debug(f"{self.name}: fault-injected -> {forced.value}")
                return forced
        return self.action(bb)


# ── fault injection ──────────────────────────────────────────────────

class FaultInjector:
    """
    Test-only hook. Lets a test force a specific named ActionNode to fail
    or hang on cue, so the tree's retry/timeout/abort behavior can be
    verified without needing real hardware to misbehave at the right
    moment. Register it on the blackboard under "fault_injector" before
    running the tree.
    """

    def __init__(self) -> None:
        self._forced_failures: set[str] = set()
        self._forced_timeouts: set[str] = set()

    def force_failure(self, node_name: str) -> None:
        self._forced_failures.add(node_name)

    def force_timeout(self, node_name: str) -> None:
        self._forced_timeouts.add(node_name)

    def clear(self) -> None:
        self._forced_failures.clear()
        self._forced_timeouts.clear()

    def check(self, node_name: str) -> Status | None:
        if node_name in self._forced_failures:
            return Status.FAILURE
        if node_name in self._forced_timeouts:
            return Status.RUNNING
        return None


# ── black-box recorder ───────────────────────────────────────────────

@dataclass
class MissionRecorder:
    """
    Flight recorder for a single artefact's mission run. One JSON line per
    ticked node: timestamp, node name, resulting status, and a blackboard
    snapshot. Enough to reconstruct exactly what the robot was doing and
    what it knew when something went wrong — for your own debugging and
    as concrete evidence of graceful failure handling if a judge asks
    "what happens when it doesn't work."
    """

    object_id: str
    output_dir: Path
    entries: list[dict] = field(default_factory=list)

    def record(self, node_name: str, status: Status, bb: Blackboard) -> None:
        self.entries.append({
            "t": time.time(),
            "node": node_name,
            "status": status.value,
            "blackboard": bb.snapshot(),
        })

    def flush(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.object_id}_blackbox.jsonl"
        with open(path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry, default=str) + "\n")
        return path


def run_to_completion(root: Node, bb: Blackboard, max_ticks: int = 2000, tick_interval: float = 0.02) -> Status:
    """
    Ticks a tree until it returns SUCCESS or FAILURE, or gives up after
    max_ticks. This ceiling exists so a bug that makes every node return
    RUNNING forever produces a loud FAILURE instead of hanging the mission
    process silently forever — the exact failure mode this whole engine
    exists to prevent everywhere else.
    """
    for _ in range(max_ticks):
        status = root.tick(bb)
        if status != Status.RUNNING:
            return status
        time.sleep(tick_interval)
    logger.error(f"Tree did not resolve within {max_ticks} ticks — treating as FAILURE")
    return Status.FAILURE