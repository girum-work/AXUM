"""
fetch_age_dataset.py — split the AGE trilingual corpus into per-language text.

AGE (Amharic / Ge'ez / English) is verse-aligned: every record holds the same
verse in all three languages. That makes it the strongest possible control for
the scaling experiment. Earlier corpora matched only loosely:

    Wikipedia vs Bible      -> different genre  (confounded)
    eBible NT vs Ge'ez OT   -> different books  (confounded)
    STEPBible parallel OT   -> same books, different translations
    AGE                     -> SAME VERSES      (clean)

With semantic content held identical, any accuracy gap is a property of the
writing system and morphology, not of what the text happens to talk about.

Records: 17,516. Fields: amh, gez, eng.
Companion file Kufale.{json,csv} is the Book of Jubilees, Ge'ez only.

LICENCE: the upstream repository declares no SPDX licence. Treat as research
use, cite the authors, and confirm terms before redistributing.
Source: https://github.com/HenokB/AGE-Dataset

Usage:
    python scripts/fetch_age_dataset.py --field gez --language geez \\
        --output-dir data/corpus_raw/geez_age
    python scripts/fetch_age_dataset.py --field amh --language amharic \\
        --output-dir data/corpus_raw/amharic_age
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

RAW_URL = "https://raw.githubusercontent.com/HenokB/AGE-Dataset/main/{name}.json"
REPO_URL = "https://github.com/HenokB/AGE-Dataset"
USER_AGENT = "AXUM-Rover-Corpus/1.0 (heritage OCR research)"

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if ETHIOPIC_RE.match(c)) / len(dense)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="AGE", choices=["AGE", "Kufale"])
    parser.add_argument("--field", required=True, help="Record key, e.g. gez or amh")
    parser.add_argument("--language", required=True, help="Label, e.g. geez")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-ethiopic-ratio", type=float, default=0.6)
    parser.add_argument("--lines-per-file", type=int, default=200)
    args = parser.parse_args()

    url = RAW_URL.format(name=args.dataset)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print(f"Downloading {url}")
    try:
        response = session.get(url, timeout=300)
        response.raise_for_status()
        records = json.loads(response.text)
    except (requests.RequestException, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    if not isinstance(records, list) or not records:
        print("Unexpected dataset shape", file=sys.stderr)
        return 1
    if args.field not in records[0]:
        print(f"Field '{args.field}' not in {list(records[0])}", file=sys.stderr)
        return 1
    print(f"  {len(records):,} records")

    lines: list[str] = []
    skipped = 0
    for record in records:
        value = record.get(args.field)
        if not isinstance(value, str):
            skipped += 1
            continue
        line = " ".join(value.split())
        if not line or not ETHIOPIC_RE.search(line):
            skipped += 1
            continue
        if ethiopic_ratio(line) < args.min_ethiopic_ratio:
            skipped += 1
            continue
        lines.append(line)

    if not lines:
        print("No Ethiopic lines recovered", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.dataset.lower()
    written = 0
    for start in range(0, len(lines), args.lines_per_file):
        part = lines[start:start + args.lines_per_file]
        path = args.output_dir / f"{args.language}_{tag}_{written:04d}.txt"
        path.write_text("\n".join(part), encoding="utf-8")
        written += 1

    (args.output_dir / "LICENCE.txt").write_text(
        f"Source: AGE-Dataset ({args.dataset}.json), field '{args.field}'\n"
        f"Repository: {REPO_URL}\n"
        f"Licence: none declared upstream. Research use; cite the authors and\n"
        f"confirm terms before redistribution.\n",
        encoding="utf-8",
    )

    total = sum(len(ETHIOPIC_RE.findall(line)) for line in lines)
    print(f"{args.language}: {len(lines):,} lines ({skipped:,} skipped) "
          f"across {written} files")
    print(f"  Ethiopic characters: {total:,}")
    print(f"\nNEXT: python scripts/prepare_restoration_corpus.py "
          f"--input-dir {args.output_dir} "
          f"--output data/restoration_corpus_{args.language}_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
