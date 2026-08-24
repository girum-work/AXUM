"""
fetch_stepbible_corpus.py — pull Ethiopic scripture from the STEPBible REST API.

WHY THIS EXISTS: the Ge'ez module (fetch_sword_bible.py) covers only the
Octateuch plus Psalms/Proverbs. eBible's Amharic package is New Testament only,
so the two never overlap and any comparison confounds language with book.
STEPBible carries a full Amharic Bible, so the default book list below is
exactly the Ge'ez book set -- same content, different language, which is the
only way to isolate the language effect.

API notes (undocumented; established by probing):
    /rest/bible/text/{version}/{ref}          -> always errors
    /rest/bible/getBibleText/{version}/{ref}  -> works
Response carries HTML in `value` and a `nextChapter.osisKeyId`, so chapter
counts never need hardcoding: walk until the book code changes.

Requests are serialised with a delay -- this is a small charity-run server.

Usage:
    python scripts/fetch_stepbible_corpus.py --version AmhNASV --language amharic \\
        --output-dir data/corpus_raw/amharic_bible_step
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path

import requests

API_URL = "https://www.stepbible.org/rest/bible/getBibleText/{version}/{ref}"
VERSION_URL = "https://www.stepbible.org/version.jsp?version={version}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# OSIS codes for the books present in the Ge'ez module.
GEEZ_PARALLEL_BOOKS = (
    "Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth", "Ps", "Prov",
)

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
TAG_RE = re.compile(r"<[^>]+>")
MAX_ATTEMPTS = 4


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if ETHIOPIC_RE.match(c)) / len(dense)


def clean_html(value: str, min_ratio: float) -> list[str]:
    """
    Strip STEPBible markup and keep genuinely Ethiopic lines.

    Args:
        value: HTML fragment from the API `value` field
        min_ratio: Per-line Ethiopic density threshold

    Returns:
        Cleaned lines
    """
    text = html.unescape(TAG_RE.sub("\n", value))
    lines: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        # Drop verse numbers left behind once tags are gone.
        line = re.sub(r"(?<![\u1200-\u137F])\d+(?![\u1200-\u137F])", " ", line)
        line = " ".join(line.split())
        if not line or not ETHIOPIC_RE.search(line):
            continue
        if ethiopic_ratio(line) < min_ratio:
            continue
        lines.append(line)
    return lines


def fetch_chapter(session: requests.Session, version: str, ref: str) -> dict | None:
    """
    Fetch one chapter with backoff.

    Args:
        session: Shared HTTP session
        version: STEPBible version id
        ref: OSIS reference, e.g. "Gen.1"

    Returns:
        Parsed JSON, or None if the chapter is unavailable
    """
    url = API_URL.format(version=version, ref=ref)
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 429:
                time.sleep(2 ** attempt * 3)
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("errorMessage"):
                return None
            return payload
        except (requests.RequestException, ValueError):
            if attempt == MAX_ATTEMPTS - 1:
                return None
            time.sleep(2 ** attempt * 2)
    return None


def fetch_book(session: requests.Session, version: str, book: str,
               min_ratio: float, delay: float) -> list[str]:
    """
    Walk every chapter of a book via nextChapter until the book changes.

    Args:
        session: Shared HTTP session
        version: STEPBible version id
        book: OSIS book code
        min_ratio: Ethiopic density threshold
        delay: Seconds between requests

    Returns:
        Cleaned lines for the whole book
    """
    lines: list[str] = []
    ref = f"{book}.1"
    chapters = 0
    while ref and ref.split(".")[0] == book:
        payload = fetch_chapter(session, version, ref)
        if payload is None:
            break
        lines.extend(clean_html(payload.get("value", ""), min_ratio))
        chapters += 1
        ref = (payload.get("nextChapter") or {}).get("osisKeyId") or ""
        time.sleep(delay)
    print(f"  {book:<5} {chapters:>3} chapters | {len(lines):>5,} lines")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="STEPBible id, e.g. AmhNASV")
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--books", nargs="*", default=list(GEEZ_PARALLEL_BOOKS),
                        help="OSIS codes; defaults to the Ge'ez book set")
    parser.add_argument("--min-ethiopic-ratio", type=float, default=0.6)
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between requests")
    parser.add_argument("--lines-per-file", type=int, default=200)
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    print(f"Fetching {args.version}: {len(args.books)} books")
    lines: list[str] = []
    for book in args.books:
        lines.extend(fetch_book(session, args.version, book,
                                args.min_ethiopic_ratio, args.delay))

    if not lines:
        print("No Ethiopic lines recovered", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(lines), args.lines_per_file):
        part = lines[start:start + args.lines_per_file]
        path = args.output_dir / f"{args.language}_{args.version.lower()}_{written:04d}.txt"
        path.write_text("\n".join(part), encoding="utf-8")
        written += 1

    licence = ""
    try:
        page = session.get(VERSION_URL.format(version=args.version), timeout=60).text
        blurb = TAG_RE.sub(" ", page)
        licence = " ".join(blurb.split())[:1500]
    except requests.RequestException:
        pass
    (args.output_dir / "LICENCE.txt").write_text(
        f"Source: STEPBible version '{args.version}'\n"
        f"URL: {VERSION_URL.format(version=args.version)}\n\n{licence}\n",
        encoding="utf-8",
    )

    total = sum(len(ETHIOPIC_RE.findall(line)) for line in lines)
    print(f"\n{args.language}: {len(lines):,} lines across {written} files")
    print(f"  Ethiopic characters: {total:,}")
    print(f"  Check {args.output_dir / 'LICENCE.txt'} before redistributing")
    print(f"\nNEXT: python scripts/prepare_restoration_corpus.py "
          f"--input-dir {args.output_dir} "
          f"--output data/restoration_corpus_{args.language}_parallel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
