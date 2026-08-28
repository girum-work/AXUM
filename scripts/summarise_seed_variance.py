"""
summarise_seed_variance.py — aggregate repeated training runs.

The Ge'ez/Amharic gap was measured from one run per language, so it carried an
unknown amount of seed-to-seed noise. This reads the per-run logs written by
train_restoration.py --tag and reports the spread, then tests the gap against
it with Welch's t-test (the two languages need not have equal variance or an
equal number of runs).

Usage:
    python scripts/summarise_seed_variance.py
    python scripts/summarise_seed_variance.py --logs logs/restoration
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Return (difference, standard error, Welch-Satterthwaite dof)."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return mb - ma, 0.0, float(na + nb - 2)
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return mb - ma, se, dof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs/restoration"))
    args = parser.parse_args()

    runs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for path in sorted(args.logs.glob("*_training.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "seed" not in record:
            continue  # written before --seed was recorded; not attributable
        runs[record["language"]].append((record["seed"], record["best_top1"]))

    if not runs:
        print(f"No seed-tagged runs in {args.logs}. Train with --tag to record them.")
        return 1

    summary: dict[str, list[float]] = {}
    for language, entries in sorted(runs.items()):
        scores = [score for _, score in sorted(entries)]
        summary[language] = scores
        n = len(scores)
        mean = sum(scores) / n
        sd = (math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
              if n > 1 else float("nan"))
        seeds = ", ".join(str(seed) for seed, _ in sorted(entries))
        print(f"{language:<10} n={n}  mean {mean:.2%}  sd {sd:.2%}  "
              f"min {min(scores):.2%}  max {max(scores):.2%}")
        print(f"{'':<10} seeds: {seeds}")

    usable = {lang: s for lang, s in summary.items() if len(s) > 1}
    if len(usable) == 2:
        (la, sa), (lb, sb) = sorted(usable.items())
        diff, se, dof = welch(sa, sb)
        half = 1.96 * se  # normal approximation; dof is reported for judgement
        print(f"\n{lb} - {la}: {diff:+.2%}  95% CI [{diff - half:+.2%}, "
              f"{diff + half:+.2%}]  se {se:.2%}  dof {dof:.1f}")
        print("Gap is within seed noise." if abs(diff) < half
              else "Gap exceeds seed noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
