"""
fetch_amharic_news.py — Amharic news corpus from Abebe et al.'s classification set.

51,483 articles, 54,006,405 Ethiopic characters in the `article` column. This is
the largest cleanly-licensed Amharic corpus located so far (MIT), which matters
because the other large sources cannot be redistributed:

    AGE-Dataset      no licence declared
    CrossWire Ge'ez  non-commercial only
    geezorg/data     NOASSERTION
    this             MIT

Its value to the scaling experiment is reach. The Wikipedia curve topped out at
7.7M characters; this extends the same measurement roughly sevenfold, which is
what makes it possible to ask whether the accuracy gains keep arriving or flatten.

News is its own genre, so tag runs `--domain news`. Do not compare these numbers
against scripture runs -- that mixes genre into a size measurement, which is the
confound this experiment exists to avoid.

Source: https://github.com/IsraelAbebe/An-Amharic-News-Text-classification-Dataset

Usage:
    python scripts/fetch_amharic_news.py --output-dir data/corpus_raw/amharic_news
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

ZIP_URL = (
    "https://raw.githubusercontent.com/IsraelAbebe/"
    "An-Amharic-News-Text-classification-Dataset/main/data/Amharic%20News%20Dataset.zip"
)
REPO_URL = "https://github.com/IsraelAbebe/An-Amharic-News-Text-classification-Dataset"
USER_AGENT = "AXUM-Rover-Corpus/1.0 (heritage OCR research)"

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")

# The CSV holds full article bodies, so the field limit needs raising.
csv.field_size_limit(10_000_000)


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if ETHIOPIC_RE.match(c)) / len(dense)


def clean_article(text: str, min_ratio: float) -> list[str]:
    """
    Split an article body into usable Ethiopic lines.

    Args:
        text: Raw article text
        min_ratio: Per-line Ethiopic density threshold

    Returns:
        Cleaned lines
    """
    lines: list[str] = []
    for chunk in re.split(r"[\n\r]+|(?<=\u1362)", text):
        chunk = " ".join(chunk.split())
        if len(chunk) < 20 or not ETHIOPIC_RE.search(chunk):
            continue
        if ethiopic_ratio(chunk) < min_ratio:
            continue
        lines.append(chunk)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", default="article", choices=["article", "headline"])
    parser.add_argument("--max-articles", type=int, default=0, help="0 = all")
    parser.add_argument("--min-ethiopic-ratio", type=float, default=0.7)
    parser.add_argument("--lines-per-file", type=int, default=2000)
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print(f"Downloading {ZIP_URL}")
    try:
        response = session.get(ZIP_URL, timeout=600)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(response.content) / 1024 / 1024:.1f} MB")

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    raw = archive.read(member).decode("utf-8", "ignore")

    lines: list[str] = []
    used = 0
    for record in csv.DictReader(io.StringIO(raw)):
        value = record.get(args.field)
        if not isinstance(value, str) or not value.strip():
            continue
        found = clean_article(value, args.min_ethiopic_ratio)
        if not found:
            continue
        lines.extend(found)
        used += 1
        if args.max_articles and used >= args.max_articles:
            break

    if not lines:
        print("No Ethiopic lines recovered", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(lines), args.lines_per_file):
        part = lines[start:start + args.lines_per_file]
        path = args.output_dir / f"amharic_news_{written:04d}.txt"
        path.write_text("\n".join(part), encoding="utf-8")
        written += 1

    (args.output_dir / "LICENCE.txt").write_text(
        f"Source: An Amharic News Text classification Dataset\n"
        f"Repository: {REPO_URL}\nLicence: MIT\n"
        f"Cite: Azime & Mohammed, 'An Amharic News Text classification Dataset', 2021\n",
        encoding="utf-8",
    )

    total = sum(len(ETHIOPIC_RE.findall(line)) for line in lines)
    print(f"amharic news: {used:,} articles -> {len(lines):,} lines "
          f"across {written} files")
    print(f"  Ethiopic characters: {total:,}")
    print(f"\nNEXT: python scripts/prepare_restoration_corpus.py "
          f"--input-dir {args.output_dir} "
          f"--output data/restoration_corpus_amharic_news.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
