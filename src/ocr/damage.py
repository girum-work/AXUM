"""
AXUM ROVER — Synthetic Ge'ez inscription damage
=================================================
Simulates stone erosion, OCR confusion, and vowel-mark loss for
LLM restoration training data. One [MISSING] token = one grapheme cluster.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum

try:
    import regex as re_grapheme
except ImportError:
    re_grapheme = None  # type: ignore

# Ethiopic script + word space (U+1361) + common punctuation
GEEZ_CHAR_RE = re.compile(
    r"[\u1200-\u137F\u1360-\u1368\u1369-\u137C]"
)

# Combining marks often lost on weathered stone.
# Only U+135D-U+135F are category Mn. U+1360-U+1368 are punctuation (Po) and
# U+1369-U+137C are numerals (No); including them glued marks onto the preceding
# syllable, so "\u120D\u1362" tokenised as one unit instead of two.
COMBINING_MARK_RE = re.compile(r"[\u135D-\u135F]")

# Visually similar fidels for OCR-style substitution (before erasure)
OCR_CONFUSIONS: dict[str, list[str]] = {
    "ሐ": ["ሓ", "ሔ", "ሕ"],
    "ላ": ["ሌ", "ል"],
    "ማ": ["ሜ", "ም"],
    "ሰ": ["ሱ", "ሲ"],
    "ቅ": ["ቆ", "ቇ"],
    "ነ": ["ኑ", "ኒ"],
    "አ": ["ኡ", "ኢ"],
    "ዓ": ["ዔ", "ዕ"],
    "ገ": ["ጉ", "ጊ"],
}


class DamageMode(str, Enum):
    """Damage simulation strategies."""

    ERASURE = "erasure"
    EDGE_EROSION = "edge_erosion"
    OCR_CONFUSION = "ocr_confusion"
    VOWEL_DROP = "vowel_drop"
    SPAN_LOSS = "span_loss"


# Known-length gap: the reader can count the lost graphemes.
MISSING_MARKER = "[MISSING]"
# Unknown-length gap: a break where even the extent is lost. Aeneas keeps these
# separate because predicting into an unmeasured gap is a harder task.
UNKNOWN_GAP_MARKER = "[GAP]"

# Main Ethiopic syllabary: consonant = (cp - 0x1200) // 8, vowel = remainder.
SYLLABARY_START = 0x1200
SYLLABARY_END = 0x1357


@dataclass
class DamageResult:
    """Output of one damage application."""

    damaged: str
    mode: DamageMode
    damage_rate: float
    missing_count: int
    unknown_gap_count: int = 0


def split_graphemes(text: str) -> list[str]:
    """
    Split text into extended grapheme clusters (handles combining marks).

    Args:
        text: Ge'ez string (spaces preserved as their own tokens)

    Returns:
        List of grapheme cluster strings
    """
    if not text:
        return []

    if re_grapheme is not None:
        return re_grapheme.findall(r"\X", text)

    # Fallback: treat space separately, else single codepoints + following marks
    clusters: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            clusters.append(" ")
            i += 1
            continue
        j = i + 1
        while j < len(text) and COMBINING_MARK_RE.match(text[j]):
            j += 1
        clusters.append(text[i:j])
        i = j
    return clusters


def join_graphemes(clusters: list[str]) -> str:
    """Rejoin grapheme list to a string."""
    return "".join(clusters)


def _edge_weight(index: int, total: int, word_start: int, word_end: int) -> float:
    """
    Higher weight at word edges (exposed on stone).

    Args:
        index: Grapheme index in full text
        total: Total grapheme count
        word_start: Index of first grapheme in current word
        word_end: Index after last grapheme in current word
    """
    if total <= 1:
        return 1.0
    dist_start = index - word_start
    dist_end = word_end - 1 - index
    edge = min(dist_start, dist_end)
    if edge <= 0:
        return 2.2
    if edge == 1:
        return 1.5
    return 1.0


def _apply_erasure(
    clusters: list[str],
    rate: float,
    edge_bias: bool = False,
) -> list[str]:
    """Replace graphemes with [MISSING] (one token per grapheme)."""
    out: list[str] = []
    word_start = 0
    i = 0
    n = len(clusters)

    while i < n:
        g = clusters[i]
        if g == " ":
            out.append(g)
            i += 1
            word_start = i
            continue

        word_end = i
        while word_end < n and clusters[word_end] != " ":
            word_end += 1

        while i < word_end:
            w = _edge_weight(i, n, word_start, word_end) if edge_bias else 1.0
            if random.random() < min(0.95, rate * w):
                out.append("[MISSING]")
            else:
                out.append(clusters[i])
            i += 1
        word_start = i

    return out


def _apply_ocr_confusion(clusters: list[str], rate: float) -> list[str]:
    """Substitute similar fidels, then optionally erase."""
    out: list[str] = []
    for g in clusters:
        if g == " ":
            out.append(g)
            continue
        base = g[0] if g else ""
        if base in OCR_CONFUSIONS and random.random() < rate * 0.6:
            sub = random.choice(OCR_CONFUSIONS[base])
            out.append(sub + g[1:] if len(g) > 1 else sub)
        elif random.random() < rate * 0.3:
            out.append("[MISSING]")
        else:
            out.append(g)
    return out


def _strip_vowel(cluster: str) -> str:
    """
    Reduce a fidel to its base consonant (vowel order 0).

    Ethiopic encodes the vowel in the syllable itself, not as a combining mark,
    so weathering that erases the vowel stroke yields the base form rather than
    a bare consonant plus a lost mark.

    Args:
        cluster: One grapheme cluster

    Returns:
        The order-0 form, or the cluster unchanged if it is not a syllable
    """
    if not cluster:
        return cluster
    code = ord(cluster[0])
    if SYLLABARY_START <= code <= SYLLABARY_END:
        return chr(code - (code - SYLLABARY_START) % 8) + cluster[1:]
    return cluster


def _apply_vowel_drop(clusters: list[str], rate: float) -> list[str]:
    """Erode vowel strokes, leaving the base consonant legible."""
    out: list[str] = []
    for g in clusters:
        if g == " ":
            out.append(g)
            continue
        if random.random() < rate:
            stripped = _strip_vowel(g)
            out.append(stripped if stripped != g else MISSING_MARKER)
        elif random.random() < rate * 0.15:
            out.append(MISSING_MARKER)
        else:
            out.append(g)
    return out


def _apply_span_loss(
    clusters: list[str],
    rate: float,
    geometric_p: float = 0.1,
    unknown_gap_prob: float = 0.25,
) -> list[str]:
    """
    Remove contiguous runs rather than scattered graphemes.

    Real damage is spatial: a chip or crack takes out neighbours together.
    Independent per-grapheme dropout leaves an unrealistically easy context
    where survivors almost always flank each loss.

    Args:
        clusters: Grapheme clusters
        rate: Target fraction of graphemes to remove
        geometric_p: Smaller values produce longer spans
        unknown_gap_prob: Chance a span collapses to one unknown-length gap

    Returns:
        Damaged clusters
    """
    out = list(clusters)
    total = sum(1 for g in clusters if g != " ")
    if total == 0:
        return out

    target = int(round(total * rate))
    removed = 0
    attempts = 0
    while removed < target and attempts < 100:
        attempts += 1
        length = min(np_geometric(geometric_p), max(1, total - removed))
        start = random.randrange(len(out))
        end = min(start + length, len(out))
        span = [i for i in range(start, end) if out[i] not in (" ", MISSING_MARKER,
                                                              UNKNOWN_GAP_MARKER)]
        if not span:
            continue
        if random.random() < unknown_gap_prob:
            for i in span:
                out[i] = ""
            out[span[0]] = UNKNOWN_GAP_MARKER
        else:
            for i in span:
                out[i] = MISSING_MARKER
        removed += len(span)

    return [g for g in out if g != ""]


def np_geometric(p: float) -> int:
    """Geometric sample (support >= 1) without pulling in numpy."""
    u = random.random()
    if p <= 0:
        return 1
    from math import log
    return max(1, int(log(1.0 - u) / log(1.0 - p)) + 1)


def apply_damage(
    text: str,
    damage_rate: float | None,
    mode: DamageMode | None = None,
    seed: int | None = None,
    unknown_gap_prob: float = 0.25,
) -> DamageResult:
    """
    Apply one damage simulation to clean Ge'ez text.

    Args:
        text: Clean inscription string
        damage_rate: Target fraction of graphemes affected; None samples
            uniformly in [0, 0.75] so the model sees the full severity range
            rather than one fixed difficulty
        mode: Force a mode; None picks weighted random
        seed: Optional RNG seed for reproducibility
        unknown_gap_prob: Chance a lost span collapses to an unknown-length gap

    Returns:
        DamageResult with damaged string and metadata
    """
    if seed is not None:
        random.seed(seed)

    if damage_rate is None:
        damage_rate = random.uniform(0.0, 0.75)

    if mode is None:
        mode = random.choices(
            list(DamageMode),
            weights=[35, 15, 15, 10, 25],
            k=1,
        )[0]

    clusters = split_graphemes(text)
    if not clusters:
        return DamageResult(text, mode, damage_rate, 0)

    if mode == DamageMode.ERASURE:
        damaged_clusters = _apply_erasure(clusters, damage_rate, edge_bias=False)
    elif mode == DamageMode.EDGE_EROSION:
        damaged_clusters = _apply_erasure(clusters, damage_rate, edge_bias=True)
    elif mode == DamageMode.OCR_CONFUSION:
        damaged_clusters = _apply_ocr_confusion(clusters, damage_rate)
    elif mode == DamageMode.SPAN_LOSS:
        damaged_clusters = _apply_span_loss(
            clusters, damage_rate, unknown_gap_prob=unknown_gap_prob
        )
    else:
        damaged_clusters = _apply_vowel_drop(clusters, damage_rate)

    damaged = join_graphemes(damaged_clusters)
    missing_count = damaged.count(MISSING_MARKER)

    # Ensure at least one damage for training (short texts)
    if missing_count == 0 and UNKNOWN_GAP_MARKER not in damaged \
            and len(text) > 1 and damage_rate > 0:
        mid = len(clusters) // 2
        idx = mid
        while idx < len(clusters) and clusters[idx] == " ":
            idx += 1
        if idx < len(clusters):
            damaged_clusters = list(clusters)
            damaged_clusters[idx] = MISSING_MARKER
            damaged = join_graphemes(damaged_clusters)

    return DamageResult(
        damaged=damaged,
        mode=mode,
        damage_rate=damage_rate,
        missing_count=damaged.count(MISSING_MARKER),
        unknown_gap_count=damaged.count(UNKNOWN_GAP_MARKER),
    )


def confidence_from_damage(missing_count: int, total_graphemes: int) -> float:
    """
    Deterministic confidence from damage severity (for synthetic labels).

    Args:
        missing_count: Number of [MISSING] tokens
        total_graphemes: Non-space grapheme count in clean text

    Returns:
        Float in [0.35, 0.95]
    """
    if total_graphemes <= 0:
        return 0.5
    ratio = missing_count / total_graphemes
    return round(max(0.35, 0.95 - ratio * 1.1), 2)


def needs_expert_from_damage(missing_count: int, total_graphemes: int) -> bool:
    """
    Flag heavy damage for expert review (matches training labels).

    Args:
        missing_count: [MISSING] token count
        total_graphemes: Non-space grapheme count

    Returns:
        True if more than 40% of graphemes are missing
    """
    if total_graphemes <= 0:
        return True
    return (missing_count / total_graphemes) > 0.40


def count_content_graphemes(text: str) -> int:
    """Count graphemes excluding spaces."""
    return sum(1 for g in split_graphemes(text) if g != " ")


def is_valid_geez_text(text: str, min_len: int = 2, max_len: int = 120) -> bool:
    """
    Check text is mostly Ethiopic and within length bounds.

    Args:
        text: Candidate clean label
        min_len: Minimum grapheme count (non-space)
        max_len: Maximum grapheme count

    Returns:
        True if suitable for restoration corpus
    """
    if not text or not text.strip():
        return False
    geez_chars = len(GEEZ_CHAR_RE.findall(text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return False
    if geez_chars / total < 0.85:
        return False
    n = count_content_graphemes(text)
    return min_len <= n <= max_len
