"""
fetch_ebible_corpus.py — extract Ethiopic text from an eBible.org USFM package.

Companion to fetch_sword_bible.py. Same goal, different container: eBible
publishes USFM (plain-text markup), CrossWire publishes compressed zText.
Both give verified UTF-8, which is why neither route needs PDFs.

USFM markers handled:
    \\f ... \\f*   footnotes      -- dropped, they are editorial apparatus
    \\x ... \\x*   cross refs     -- dropped, mostly Latin book abbreviations
    \\w word|lemma\\w*            -- lemma stripped, surface form kept
    \\c \\v \\p \\q1 ...           -- structural markers, removed
Verse numbers are dropped so the output is running text, matching the shape
prepare_restoration_corpus.py expects.

Usage:
    python scripts/fetch_ebible_corpus.py --translation amh --language amharic \\
        --output-dir data/corpus_raw/amharic_bible
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

PACKAGE_URL = "https://ebible.org/Scriptures/{translation}_usfm.zip"
CATALOGUE_URL = "https://ebible.org/Scriptures/translations.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
NOTE_RE = re.compile(r"\\(f|x|fe)\b.*?\\\1\*", re.DOTALL)
WORD_ATTR_RE = re.compile(r"\\w\s+([^|\\]+)(?:\|[^\\]*)?\\w\*")
MARKER_RE = re.compile(r"\\[a-zA-Z]+\d*\*?")
VERSE_NUM_RE = re.compile(r"(?<![\u1200-\u137F])\d+(?![\u1200-\u137F])")


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if ETHIOPIC_RE.match(c)) / len(dense)


def usfm_to_lines(raw: str, min_ratio: float) -> list[str]:
    """
    Convert one USFM book into clean Ethiopic lines.

    Args:
        raw: USFM file contents
        min_ratio: Per-line Ethiopic density threshold

    Returns:
        Cleaned lines of running text
    """
    text = NOTE_RE.sub(" ", raw)
    text = WORD_ATTR_RE.sub(r"\1", text)

    lines: list[str] = []
    for line in text.splitlines():
        if not line.startswith("\\v ") and not line.startswith("\\q") and "\\v " not in line:
            # Keep only verse-bearing lines; headers and metadata are Latin.
            if not ETHIOPIC_RE.search(line):
                continue
        line = MARKER_RE.sub(" ", line)
        line = VERSE_NUM_RE.sub(" ", line)
        line = " ".join(line.split())
        if not line or not ETHIOPIC_RE.search(line):
            continue
        if ethiopic_ratio(line) < min_ratio:
            continue
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation", required=True, help="eBible id, e.g. amh")
    parser.add_argument("--language", required=True, help="Label, e.g. amharic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-ethiopic-ratio", type=float, default=0.6)
    parser.add_argument("--lines-per-file", type=int, default=200)
    args = parser.parse_args()

    url = PACKAGE_URL.format(translation=args.translation)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print(f"Downloading {url}")
    try:
        response = session.get(url, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    print(f"  {len(response.content) / 1024:.0f} KB")

    books = [n for n in archive.namelist() if n.lower().endswith((".usfm", ".sfm"))]
    lines: list[str] = []
    for name in sorted(books):
        raw = archive.read(name).decode("utf-8", "ignore")
        lines.extend(usfm_to_lines(raw, args.min_ethiopic_ratio))
    print(f"  {len(books)} books parsed")

    if not lines:
        print("No Ethiopic lines recovered", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(lines), args.lines_per_file):
        part = lines[start:start + args.lines_per_file]
        path = args.output_dir / f"{args.language}_{args.translation}_{written:04d}.txt"
        path.write_text("\n".join(part), encoding="utf-8")
        written += 1

    licence = ""
    for name in archive.namelist():
        if "copyright" in name.lower() or name.lower().endswith("copr.htm"):
            licence = re.sub(r"<[^>]+>", " ", archive.read(name).decode("utf-8", "ignore"))
            licence = " ".join(licence.split())
            break
    (args.output_dir / "LICENCE.txt").write_text(
        f"Source: eBible.org translation '{args.translation}'\nURL: {url}\n"
        f"Catalogue: {CATALOGUE_URL}\n\n{licence}\n",
        encoding="utf-8",
    )

    total = sum(len(ETHIOPIC_RE.findall(line)) for line in lines)
    print(f"{args.language}: {len(lines):,} lines across {written} files")
    print(f"  Ethiopic characters: {total:,}")
    print(f"\nNEXT: python scripts/prepare_restoration_corpus.py "
          f"--input-dir {args.output_dir} "
          f"--output data/restoration_corpus_{args.language}_bible.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
