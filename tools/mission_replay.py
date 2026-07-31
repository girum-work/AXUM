"""
AXUM ROVER - Black-box mission replay viewer.

WHAT: Reads a MissionRecorder .jsonl file (written by mission_tree.py's
run_artefact_mission()) and turns it into a readable report — a timeline
of every phase, how long each took, which retries fired and why, and what
the blackboard looked like at each transition.

WHY standalone: this only depends on the .jsonl file format MissionRecorder
already writes (see behavior_tree.py's MissionRecorder.record()) — one
JSON object per line: {"t": <unix time>, "node": <name>, "status": <str>,
"blackboard": {...snapshot...}}. No dependency on controller.py, config.py,
or anything hardware-related, so this runs anywhere the log file can be
copied to, including a laptop with no repo checked out at all.

USAGE:
    python mission_replay.py path/to/AXUM-OBJ-001_blackbox.jsonl
    python mission_replay.py path/to/AXUM-OBJ-001_blackbox.jsonl --html report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayEntry:
    t: float
    node: str
    status: str
    blackboard: dict[str, Any]


@dataclass
class NodeTimeline:
    node: str
    entries: list[ReplayEntry] = field(default_factory=list)

    @property
    def final_status(self) -> str:
        return self.entries[-1].status if self.entries else "UNKNOWN"

    @property
    def attempt_count(self) -> int:
        return len(self.entries)

    @property
    def duration_seconds(self) -> float:
        if len(self.entries) < 2:
            return 0.0
        return self.entries[-1].t - self.entries[0].t


def load_entries(path: Path) -> list[ReplayEntry]:
    entries = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(ReplayEntry(t=data["t"], node=data["node"], status=data["status"], blackboard=data.get("blackboard", {})))
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"WARNING: skipping malformed line {line_number}: {exc}", file=sys.stderr)
    return entries


def group_by_node(entries: list[ReplayEntry]) -> list[NodeTimeline]:
    """
    Groups entries by node name IN FIRST-SEEN ORDER (not alphabetical) so
    the report reads as a mission timeline, not a shuffled index. A node
    can appear multiple times (each retry attempt is a separate tick).
    """
    order: list[str] = []
    grouped: dict[str, NodeTimeline] = {}
    for entry in entries:
        if entry.node not in grouped:
            grouped[entry.node] = NodeTimeline(node=entry.node)
            order.append(entry.node)
        grouped[entry.node].entries.append(entry)
    return [grouped[name] for name in order]


def find_last_abort_reason(entries: list[ReplayEntry]) -> str | None:
    for entry in reversed(entries):
        reason = entry.blackboard.get("last_abort_reason") or entry.blackboard.get("last_invariant_violation")
        if reason:
            return reason
    return None


def render_text_report(entries: list[ReplayEntry], object_id: str) -> str:
    if not entries:
        return f"No entries found for {object_id} — empty or unreadable log file."

    timelines = group_by_node(entries)
    start_t = entries[0].t
    end_t = entries[-1].t
    lines = []
    lines.append(f"=== AXUM Mission Replay: {object_id} ===")
    lines.append(f"Total duration: {end_t - start_t:.2f}s across {len(entries)} recorded ticks, {len(timelines)} distinct phases")
    lines.append("")
    lines.append(f"{'PHASE':<20} {'ATTEMPTS':<10} {'FINAL STATUS':<12} {'ELAPSED':<10}")
    lines.append("-" * 55)
    for tl in timelines:
        elapsed = f"{tl.entries[0].t - start_t:+.2f}s"
        status_marker = "✓" if tl.final_status == "SUCCESS" else ("✗" if tl.final_status == "FAILURE" else "…")
        lines.append(f"{tl.node:<20} {tl.attempt_count:<10} {status_marker} {tl.final_status:<10} {elapsed:<10}")
        if tl.attempt_count > 1:
            statuses = " -> ".join(e.status for e in tl.entries)
            lines.append(f"    retries: {statuses}")

    abort_reason = find_last_abort_reason(entries)
    if abort_reason:
        lines.append("")
        lines.append(f"LAST RECORDED ABORT/VIOLATION REASON: {abort_reason}")

    overall = entries[-1].status
    lines.append("")
    lines.append(f"MISSION OUTCOME: {overall}")
    return "\n".join(lines)


def render_html_report(entries: list[ReplayEntry], object_id: str) -> str:
    timelines = group_by_node(entries)
    start_t = entries[0].t if entries else 0
    rows = []
    for tl in timelines:
        color = "#27ae60" if tl.final_status == "SUCCESS" else ("#c0392b" if tl.final_status == "FAILURE" else "#7f8c8d")
        elapsed = tl.entries[0].t - start_t
        retry_note = f" (retried {tl.attempt_count}x)" if tl.attempt_count > 1 else ""
        rows.append(
            f'<tr><td>{tl.node}</td><td style="color:{color};font-weight:bold">{tl.final_status}{retry_note}</td>'
            f'<td>+{elapsed:.2f}s</td></tr>'
        )
    abort_reason = find_last_abort_reason(entries)
    abort_html = f'<p style="color:#c0392b"><b>Last abort/violation reason:</b> {abort_reason}</p>' if abort_reason else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AXUM Mission Replay — {object_id}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; }}
table {{ width: 100%; border-collapse: collapse; }}
td, th {{ padding: 8px; border-bottom: 1px solid #ddd; text-align: left; }}
</style></head><body>
<h2>AXUM Mission Replay — {object_id}</h2>
<table><tr><th>Phase</th><th>Outcome</th><th>Elapsed</th></tr>
{"".join(rows)}
</table>
{abort_html}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="AXUM black-box mission replay viewer")
    parser.add_argument("logfile", type=Path, help="path to a *_blackbox.jsonl file")
    parser.add_argument("--html", type=Path, default=None, help="also write an HTML report to this path")
    args = parser.parse_args()

    if not args.logfile.exists():
        raise SystemExit(f"File not found: {args.logfile}")

    object_id = args.logfile.stem.replace("_blackbox", "")
    entries = load_entries(args.logfile)

    print(render_text_report(entries, object_id))

    if args.html:
        args.html.write_text(render_html_report(entries, object_id))
        print(f"\nHTML report written to {args.html}")


if __name__ == "__main__":
    main()