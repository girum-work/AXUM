"""
run_corpus_scaling.py — measure restoration accuracy as a function of corpus size.

WHAT: trains a character-level n-gram restorer on increasing subsets of a
corpus and evaluates it on held-out damaged text, producing a scaling curve.

WHY this experiment: Ge'ez and Amharic are sibling languages in the same
script, but Amharic has vastly more digitised text. Reporting one Ge'ez number
against one Amharic number would confound TWO variables -- corpus size and
language. Training Amharic against ITSELF at matched corpus sizes isolates
size as the only factor, and the Ge'ez point can then be located on that curve.
That turns "Ge'ez does worse" into "Ge'ez sits at N phrases, where the curve
predicts X% -- reaching Y% needs M phrases."

WHY an n-gram model rather than a neural one: restoring a missing fidel IS
next-character prediction from context, so a character n-gram is a genuine
restorer, not a stand-in. It trains in seconds on CPU, needs no GPU, and has
no hyperparameters to confound the comparison. When the neural restorer exists
it can be dropped into the same harness and evaluated against the same curve.

Measured corpus reality (2026-08):
    Ge'ez   (Enoch, Dillmann)      1,545 chunks /    88,297 Ethiopic chars
    Amharic (Wikipedia dump)     161,851 chunks / 6,450,820 Ethiopic chars

Usage:
    python scripts/run_corpus_scaling.py \\
        --corpus data/restoration_corpus_amharic.json --language amharic
    python scripts/run_corpus_scaling.py \\
        --corpus data/restoration_corpus_geez.json --language geez
    python scripts/run_corpus_scaling.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.damage import DamageMode, apply_damage, split_graphemes

RESULTS_DIR = Path("logs/corpus_scaling")
DEFAULT_SIZES = (250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000)
MISSING = "[MISSING]"

# Ethiopic syllables, punctuation and numerals. Everything else -- Latin,
# ASCII digits, brackets -- is corpus-specific noise, not language content.
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")


def ethiopic_only(tokens: list[str]) -> list[str]:
    """
    Drop non-Ethiopic graphemes, collapsing them to a single space.

    Args:
        tokens: Grapheme tokens from split_graphemes

    Returns:
        Tokens containing only Ethiopic graphemes and single spaces

    Without this the comparison is invalid. Amharic Wikipedia is 8.3%
    non-Ethiopic (dates, ISBNs, Latin names) against Ge'ez Enoch's 0.0%, and
    that alone inflated Amharic's grapheme vocabulary from 262 types to 1,757 --
    so measured accuracy tracked corpus breadth rather than corpus size.
    """
    out: list[str] = []
    for token in tokens:
        if ETHIOPIC_RE.match(token):
            out.append(token)
        elif out and out[-1] != " ":
            out.append(" ")
    while out and out[-1] == " ":
        out.pop()
    return out


@dataclass
class ScalingPoint:
    """One point on the corpus-size curve."""

    language: str
    domain: str
    train_phrases: int
    train_chars: int
    eval_phrases: int
    slots: int
    top1_accuracy: float
    top3_accuracy: float
    perplexity: float
    unique_fidels: int


class CharNGram:
    """
    Backoff character n-gram model over grapheme clusters.

    Args:
        order: Maximum context length in graphemes

    Stupid-backoff rather than full Kneser-Ney: with corpora spanning 1.5k to
    160k chunks the smoothing method would itself become a confound, and
    backoff behaves predictably at both extremes.
    """

    def __init__(self, order: int = 5) -> None:
        self.order = order
        self.counts: list[dict[tuple[str, ...], Counter]] = [
            defaultdict(Counter) for _ in range(order + 1)
        ]
        self.vocab: set[str] = set()
        self.total_tokens = 0

    def train(self, sequences: list[list[str]]) -> None:
        """Accumulate n-gram counts from grapheme sequences."""
        for tokens in sequences:
            padded = ["<s>"] * self.order + tokens + ["</s>"]
            self.vocab.update(tokens)
            self.total_tokens += len(tokens)
            for index in range(self.order, len(padded)):
                target = padded[index]
                for back in range(self.order + 1):
                    context = tuple(padded[index - back:index])
                    self.counts[back][context][target] += 1

    def distribution(self, context: tuple[str, ...]) -> Counter:
        """Return the highest-order non-empty distribution for a context."""
        for back in range(min(self.order, len(context)), -1, -1):
            table = self.counts[back].get(tuple(context[len(context) - back:]) if back else ())
            if table:
                return table
        return Counter()

    def rank(self, context: tuple[str, ...], limit: int) -> list[str]:
        """Most likely next graphemes given a context, best first."""
        return [token for token, _ in self.distribution(context).most_common(limit)]

    def logprob(self, context: tuple[str, ...], target: str) -> float:
        """Backoff log-probability of one grapheme, with add-one smoothing."""
        table = self.distribution(context)
        vocab_size = max(len(self.vocab), 1)
        return math.log((table.get(target, 0) + 1) / (sum(table.values()) + vocab_size))


def load_chunks(path: Path) -> list[str]:
    """Load chunk texts from a prepared restoration corpus JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("chunks", [])
    texts: list[str] = []
    for row in rows:
        text = row.get("text", "") if isinstance(row, dict) else str(row)
        if text and text.strip():
            texts.append(text.strip())
    return texts


def tokenize_damaged(damaged: str) -> list[str]:
    """
    Split damaged text into grapheme tokens, keeping [MISSING] atomic.

    Args:
        damaged: Output of apply_damage

    Returns:
        Token list aligned 1:1 with split_graphemes() of the clean phrase

    Splitting on whitespace does NOT work: split_graphemes keeps spaces as
    tokens, so a whitespace split silently misaligns every phrase containing
    a space and the evaluation set comes out empty.
    """
    tokens: list[str] = []
    for index, segment in enumerate(damaged.split(MISSING)):
        if index:
            tokens.append(MISSING)
        tokens.extend(split_graphemes(segment))
    return tokens


def ethiopic_only_keep_missing(tokens: list[str]) -> list[str]:
    """Ethiopic-only filter that preserves [MISSING] slots for alignment."""
    out: list[str] = []
    for token in tokens:
        if token == MISSING or ETHIOPIC_RE.match(token):
            out.append(token)
        elif out and out[-1] != " ":
            out.append(" ")
    while out and out[-1] == " ":
        out.pop()
    return out


def build_eval_set(
    phrases: list[str], damage_rate: float, seed: int
) -> list[tuple[list[str], list[int], list[str]]]:
    """
    Damage held-out phrases into (tokens, slot indices, gold answers).

    Args:
        phrases: Clean held-out phrases
        damage_rate: Fraction of graphemes to erase
        seed: RNG seed for reproducibility

    Returns:
        One tuple per phrase that produced at least one usable slot
    """
    rng = random.Random(seed)
    items = []
    for phrase in phrases:
        result = apply_damage(
            phrase, damage_rate, mode=DamageMode.ERASURE, seed=rng.randrange(1 << 30)
        )
        damaged_tokens = ethiopic_only_keep_missing(tokenize_damaged(result.damaged))
        gold_tokens = ethiopic_only(split_graphemes(phrase))
        if len(damaged_tokens) != len(gold_tokens):
            continue
        slots = [i for i, t in enumerate(damaged_tokens) if t == MISSING]
        if slots:
            items.append((damaged_tokens, slots, gold_tokens))
    return items


def evaluate(model: CharNGram, items) -> tuple[float, float, float, int]:
    """
    Score a model's ability to fill [MISSING] slots.

    Returns:
        (top-1 accuracy, top-3 accuracy, perplexity, slot count)
    """
    top1 = top3 = slots_seen = 0
    log_sum = 0.0
    for tokens, slots, gold in items:
        for position in slots:
            context = tuple(
                t for t in tokens[max(0, position - model.order):position]
                if t != MISSING
            )
            ranked = model.rank(context, 3)
            answer = gold[position]
            slots_seen += 1
            if ranked and ranked[0] == answer:
                top1 += 1
            if answer in ranked:
                top3 += 1
            log_sum += model.logprob(context, answer)

    if not slots_seen:
        return 0.0, 0.0, float("inf"), 0
    return (
        top1 / slots_seen,
        top3 / slots_seen,
        math.exp(-log_sum / slots_seen),
        slots_seen,
    )


def run_language(
    corpus_path: Path,
    language: str,
    domain: str,
    sizes: tuple[int, ...],
    order: int,
    damage_rate: float,
    eval_phrases: int,
    seed: int,
) -> list[ScalingPoint]:
    """Train and evaluate at each corpus size, holding everything else fixed."""
    chunks = load_chunks(corpus_path)
    rng = random.Random(seed)
    rng.shuffle(chunks)

    holdout = chunks[:eval_phrases]
    pool = chunks[eval_phrases:]
    if not pool:
        raise ValueError(f"{corpus_path} has too few chunks to split")

    items = build_eval_set(holdout, damage_rate, seed)
    if not items:
        raise ValueError(
            "Evaluation set is empty — damaged/clean tokenisation did not align"
        )
    print(f"{language}: {len(chunks):,} chunks | eval {len(items):,} phrases "
          f"| pool {len(pool):,}")

    points: list[ScalingPoint] = []
    for size in sizes:
        if size > len(pool):
            break
        subset = pool[:size]
        sequences = [ethiopic_only(split_graphemes(text)) for text in subset]
        sequences = [s for s in sequences if s]
        model = CharNGram(order=order)
        model.train(sequences)

        top1, top3, ppl, slots = evaluate(model, items)
        point = ScalingPoint(
            language=language,
            domain=domain,
            train_phrases=size,
            train_chars=sum(len(s) for s in sequences),
            eval_phrases=len(items),
            slots=slots,
            top1_accuracy=round(top1, 4),
            top3_accuracy=round(top3, 4),
            perplexity=round(ppl, 2),
            unique_fidels=len(model.vocab),
        )
        points.append(point)
        print(f"  {size:>7,} phrases | {point.train_chars:>9,} chars | "
              f"top1 {top1:6.2%} | top3 {top3:6.2%} | ppl {ppl:8.1f} | "
              f"fidels {point.unique_fidels}")

    # Always include the full pool so the largest corpus is represented even
    # when it falls between two configured sizes.
    if not points or points[-1].train_phrases < len(pool):
        sequences = [ethiopic_only(split_graphemes(text)) for text in pool]
        sequences = [s for s in sequences if s]
        model = CharNGram(order=order)
        model.train(sequences)
        top1, top3, ppl, slots = evaluate(model, items)
        point = ScalingPoint(
            language=language, domain=domain, train_phrases=len(pool),
            train_chars=sum(len(s) for s in sequences),
            eval_phrases=len(items), slots=slots,
            top1_accuracy=round(top1, 4), top3_accuracy=round(top3, 4),
            perplexity=round(ppl, 2), unique_fidels=len(model.vocab),
        )
        points.append(point)
        print(f"  {len(pool):>7,} phrases | {point.train_chars:>9,} chars | "
              f"top1 {top1:6.2%} | top3 {top3:6.2%} | ppl {ppl:8.1f} | "
              f"fidels {point.unique_fidels}  (full)")

    return points


def report() -> int:
    """Print a combined table, comparing only same-domain runs."""
    files = sorted(RESULTS_DIR.glob("*_scaling.json"))
    if not files:
        print(f"No results in {RESULTS_DIR}. Run the experiment first.")
        return 1

    print(f"{'language':9s} {'domain':13s} {'phrases':>9s} {'chars':>10s} "
          f"{'top1':>8s} {'top3':>8s} {'ppl':>8s} {'types':>7s}")
    print("-" * 78)

    by_domain: dict[str, dict[str, list[ScalingPoint]]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = [ScalingPoint(**p) for p in payload["points"]]
        domain = payload.get("domain", "unspecified")
        by_domain.setdefault(domain, {})[payload["language"]] = points
        for p in points:
            print(f"{p.language:9s} {p.domain:13s} {p.train_phrases:>9,} "
                  f"{p.train_chars:>10,} {p.top1_accuracy:>7.2%} "
                  f"{p.top3_accuracy:>7.2%} {p.perplexity:>8.1f} "
                  f"{p.unique_fidels:>7,}")

    def nearest(points: list[ScalingPoint], chars: int) -> ScalingPoint:
        """Point whose training budget is closest to `chars`."""
        return min(points, key=lambda p: abs(p.train_chars - chars))

    # Characters, not phrases, are the fair budget: chunk lengths differ
    # between corpora, so equal phrase counts are not equal amounts of text.
    print("\n" + "=" * 78)
    print("1. SCALING — more data, holding language and domain fixed")
    for domain, langs in sorted(by_domain.items()):
        for language, points in sorted(langs.items()):
            first, last = points[0], points[-1]
            growth = last.train_chars / max(first.train_chars, 1)
            gain = (last.top1_accuracy - first.top1_accuracy) * 100
            print(f"  {language:8s} {domain:13s} {growth:6.1f}x chars -> "
                  f"{gain:+.2f} pts top-1 "
                  f"({first.top1_accuracy:.2%} -> {last.top1_accuracy:.2%})")

    # Comparing a liturgical book against an encyclopedia measures genre
    # breadth, not corpus size; that confound made the first run show a LOWER
    # score for the 141x larger corpus.
    print("\n2. LANGUAGE — same domain, matched character budget")
    for domain, langs in sorted(by_domain.items()):
        if len(langs) < 2:
            continue
        budget = min(max(p.train_chars for p in pts) for pts in langs.values())
        print(f"  domain: {domain}  (budget ~{budget:,} chars)")
        ranked = sorted(langs.items(),
                        key=lambda kv: -nearest(kv[1], budget).top1_accuracy)
        for language, points in ranked:
            p = nearest(points, budget)
            print(f"    {language:8s} {p.train_chars:>9,} chars | "
                  f"top1 {p.top1_accuracy:6.2%} | top3 {p.top3_accuracy:6.2%} | "
                  f"ppl {p.perplexity:7.1f} | fidels {p.unique_fidels:>4,}")
        if len(ranked) == 2:
            (hi, hi_pts), (lo, lo_pts) = ranked
            delta = (nearest(hi_pts, budget).top1_accuracy
                     - nearest(lo_pts, budget).top1_accuracy) * 100
            print(f"    -> {hi} leads {lo} by {delta:+.2f} pts at equal data")

    print("\n3. DOMAIN — same language, does genre beat volume?")
    by_language: dict[str, dict[str, list[ScalingPoint]]] = {}
    for domain, langs in by_domain.items():
        for language, points in langs.items():
            by_language.setdefault(language, {})[domain] = points
    for language, domains in sorted(by_language.items()):
        if len(domains) < 2:
            continue
        best = max(domains.items(),
                   key=lambda kv: max(p.top1_accuracy for p in kv[1]))
        biggest = max(domains.items(),
                      key=lambda kv: max(p.train_chars for p in kv[1]))
        best_p = max(best[1], key=lambda p: p.top1_accuracy)
        big_p = max(biggest[1], key=lambda p: p.train_chars)
        print(f"  {language}: best={best[0]} ({best_p.top1_accuracy:.2%} @ "
              f"{best_p.train_chars:,} chars), "
              f"largest={biggest[0]} ({big_p.top1_accuracy:.2%} @ "
              f"{big_p.train_chars:,} chars)")
        if best[0] != biggest[0]:
            ratio = big_p.train_chars / max(best_p.train_chars, 1)
            cost = (big_p.top1_accuracy - best_p.top1_accuracy) * 100
            print(f"    -> in-domain data wins: {ratio:.1f}x MORE out-of-domain "
                  f"text scored {cost:+.2f} pts")

    unspecified = [d for d in by_domain if d == "unspecified"]
    if unspecified:
        print("\nWARNING: runs tagged 'unspecified' are not size-comparable. "
              "Re-run with --domain.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--domain", default="unspecified",
                        help="Genre label, e.g. religious or encyclopedic. "
                             "Only same-domain runs are size-comparable.")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES),
                        help="Training-set sizes in phrases; raise the ceiling "
                             "for corpora larger than the default 100k")
    parser.add_argument("--damage-rate", type=float, default=0.25)
    parser.add_argument("--eval-phrases", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true",
                        help="Print the combined table and exit")
    args = parser.parse_args()

    if args.report:
        return report()
    if not args.corpus or not args.language:
        parser.error("--corpus and --language are required unless --report")
    if not args.corpus.exists():
        print(f"Corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    points = run_language(
        args.corpus, args.language, args.domain, tuple(sorted(args.sizes)),
        args.order, args.damage_rate, args.eval_phrases, args.seed,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.language}_{args.domain}_scaling.json"
    out_path.write_text(json.dumps({
        "language": args.language,
        "domain": args.domain,
        "corpus": str(args.corpus),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ngram_order": args.order,
        "damage_rate": args.damage_rate,
        "seed": args.seed,
        "points": [asdict(p) for p in points],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
