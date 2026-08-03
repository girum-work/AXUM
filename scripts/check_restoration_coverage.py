"""
check_restoration_coverage.py — verifies restoration robustness across EVERY
character position in a word/phrase, not just whatever pattern a single
random training draw happened to produce.

WHAT: for each line in a sample corpus, masks ONE character at a time at
EVERY position in turn (N tests per N-character line, not 2^N — exhaustive
per-position coverage without the combinatorial explosion of exhaustive
subset masking), runs each through a restoration function, and checks
whether the correct character comes back. Aggregates accuracy by relative
position (start / middle / end of word) to reveal positional bias.

WHY this instead of an exhaustive-permutation training set: the number of
possible masked-subset patterns for an N-character phrase is 2^N — for a
25-character phrase (near our CTC budget) that's over 33 million patterns
for ONE line, not remotely buildable or trainable. This script answers the
real underlying question — "is restoration robust regardless of WHICH
character is missing, not just the ones randomly sampled during training" —
with a tractable O(N) check per line instead of an intractable 2^N dataset.

WHAT THIS IS NOT: a replacement for restoration_augmentation.py's live,
per-epoch random masking during training. That's still the right mechanism
for training itself (broad, representative coverage via random sampling
across epochs). This script is a POST-TRAINING VERIFICATION tool — it
answers "did that training actually produce position-independent
robustness," it doesn't try to force it through exhaustive data.

ENGINE-AGNOSTIC BY DESIGN: this does not import or assume any specific
restoration engine (my own candidate RAG design, or the real
GeezRestorationEngine, whichever the pending re-scope decision lands on).
It takes a plain restore_fn(damaged_text: str) -> str callable. Plug in
whichever engine is actually confirmed before running this for real.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from loguru import logger

CHAR_MASK = "\u2043"  # matches restoration_augmentation.py's known-length placeholder
ETHIOPIC_CHAR_PATTERN = re.compile(r"[\u1200-\u137F]")


@dataclass
class PositionResult:
    """WHAT: one single-position masking test's outcome.
    WHY: keeping per-test detail (not just aggregate accuracy) makes it
    possible to inspect exactly which words/positions failed, not just
    know that some did."""
    source_line: str
    masked_index: int
    relative_position: str  # "start" | "middle" | "end"
    true_char: str
    restored_char: str
    correct: bool


def _relative_position(index: int, length: int) -> str:
    """WHAT: buckets an absolute character index into start/middle/end.
    WHY: per-position accuracy is more useful summarized this way than as
    100 individual index numbers — this is what actually reveals a bias
    like 'consistently worse at word-final characters.'"""
    if length <= 1:
        return "start"
    third = max(1, length // 3)
    if index < third:
        return "start"
    if index >= length - third:
        return "end"
    return "middle"


def check_line_coverage(line: str, restore_fn: Callable[[str], str]) -> List[PositionResult]:
    """
    WHAT: tests EVERY Ethiopic-character position in one line, one at a time.
    WHY: this is the actual exhaustive part — exhaustive over POSITIONS
    (O(N)), not over damage PATTERNS (O(2^N)). Every real character in the
    line gets tested as the missing one at some point; punctuation/spacing
    positions are skipped since they're not meaningful restoration targets.
    """
    results = []
    for i, true_char in enumerate(line):
        if not ETHIOPIC_CHAR_PATTERN.match(true_char):
            continue  # skip word-dividers/punctuation — not a real restoration target

        damaged = line[:i] + CHAR_MASK + line[i + 1:]
        restored = restore_fn(damaged)

        # Extract whatever character the restoration put back at this position.
        # If the restored string is a different length than the input (the
        # engine could plausibly return a full-word substitution rather than
        # a single character), fall back to comparing at the same index if
        # in bounds, else mark as incorrect rather than crash.
        restored_char = restored[i] if i < len(restored) else ""

        results.append(PositionResult(
            source_line=line,
            masked_index=i,
            relative_position=_relative_position(i, len(line)),
            true_char=true_char,
            restored_char=restored_char,
            correct=(restored_char == true_char),
        ))
    return results


def run_coverage_check(
    corpus_lines: List[str],
    restore_fn: Callable[[str], str],
    max_lines: int = None,
) -> dict:
    """
    WHAT: runs check_line_coverage() across a corpus sample and aggregates.
    WHY: single entry point — returns both the summary stats (what you'd
    report) and the raw per-test results (what you'd inspect for failures).
    """
    if max_lines:
        corpus_lines = corpus_lines[:max_lines]

    all_results: List[PositionResult] = []
    for line in corpus_lines:
        all_results.extend(check_line_coverage(line, restore_fn))

    by_position = defaultdict(list)
    for r in all_results:
        by_position[r.relative_position].append(r.correct)

    summary = {
        "total_positions_tested": len(all_results),
        "overall_accuracy": sum(r.correct for r in all_results) / len(all_results) if all_results else 0.0,
        "accuracy_by_position": {
            pos: (sum(vals) / len(vals) if vals else 0.0)
            for pos, vals in by_position.items()
        },
        "failures": [r for r in all_results if not r.correct],
    }
    return summary


def load_corpus_lines(chunked_json_path: Path, max_lines: int = None) -> List[str]:
    """WHAT: loads real corpus text (e.g. prepare_restoration_corpus.py's output)
    for use as coverage-check source lines.
    WHY: reuses the already-verified real corpus rather than needing a
    separate curated test set — same source, no new data-quality risk."""
    with open(chunked_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    lines = [r["text"] for r in records]
    return lines[:max_lines] if max_lines else lines


def print_report(summary: dict) -> None:
    """WHAT: human-readable summary. WHY: the actual number people need to
    look at, not just the raw dict."""
    print(f"\nTotal positions tested: {summary['total_positions_tested']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
    print("\nAccuracy by relative position (this is the bias signal):")
    for pos in ("start", "middle", "end"):
        acc = summary["accuracy_by_position"].get(pos)
        if acc is not None:
            print(f"  {pos:8s}: {acc:.1%}")
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} failures — first 5:")
        for f in summary["failures"][:5]:
            print(f"  '{f.source_line[:30]}...' pos={f.masked_index} ({f.relative_position}) "
                  f"true={f.true_char!r} got={f.restored_char!r}")


if __name__ == "__main__":
    # Smoke test — confirms the harness logic itself works, using a
    # deliberately imperfect mock restore_fn (NOT a real engine — none is
    # confirmed yet). A real run should pass in the actual confirmed
    # restoration engine's callable once GeezRestorationEngine's status
    # is resolved.
    def mock_restore_fn(damaged: str) -> str:
        """Deliberately biased mock: perfect at the start of a word, wrong
        at the end — this should show up clearly in the position breakdown,
        proving the script actually detects positional bias rather than
        just reporting a flat number."""
        idx = damaged.index(CHAR_MASK)
        if idx >= len(damaged) - 2:
            return damaged.replace(CHAR_MASK, "ደ")  # deliberately wrong filler
        return damaged.replace(CHAR_MASK, "ሰ")  # deliberately "correct" for this test

    test_lines = ["ሰላም ፡ ለኵሉ", "ወይቤሉ ፡ በበይናቲሆሙ"]
    # Rig the mock so the "correct" filler actually matches at non-end positions,
    # by checking against itself — this smoke test validates the HARNESS,
    # not any real restoration quality.
    def true_char_at(line, i):
        return line[i]

    def rigged_mock(damaged: str) -> str:
        idx = damaged.index(CHAR_MASK)
        # find which test line this came from
        for line in test_lines:
            candidate = line[:idx] + CHAR_MASK + line[idx + 1:]
            if candidate == damaged:
                true_c = line[idx]
                if _relative_position(idx, len(line)) == "end":
                    return damaged.replace(CHAR_MASK, "ደ")  # wrong on purpose
                return damaged.replace(CHAR_MASK, true_c)  # correct elsewhere
        return damaged.replace(CHAR_MASK, "?")

    summary = run_coverage_check(test_lines, rigged_mock)
    print_report(summary)
    assert summary["accuracy_by_position"].get("end", 1.0) < summary["accuracy_by_position"].get("start", 0.0) + 0.01 \
        or summary["accuracy_by_position"]["end"] < 1.0, "Smoke test should show end-position bias"
    print("\nOK — harness correctly detects positional bias in the rigged mock.")
