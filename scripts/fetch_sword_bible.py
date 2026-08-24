"""
fetch_sword_bible.py — extract Ethiopic text from a CrossWire SWORD module.

WHAT: downloads a SWORD zText module, decompresses its blocks, strips OSIS
markup, and writes plain-text files ready for prepare_restoration_corpus.py.

WHY SWORD rather than PDF: Ethiopic PDF text layers are unreliable. Font
subsetting and custom encodings routinely yield wrong codepoints or decomposed
fidels, and the failure is SILENT -- the text looks plausible and is wrong.
That is disqualifying here, because a restoration model trained on quietly
corrupted fidels learns the corruption. SWORD modules store verified UTF-8.

Format notes (zText), which are not documented in one obvious place:
    .bzs  block index   -- 12-byte records: uint32 offset, compressed size,
                           uncompressed size
    .bzz  payload       -- zlib-compressed blocks at those offsets
    .bzv  verse index   -- per-verse locations; not needed when the goal is
                           the text itself rather than verse addressing

Known module:
    Geez  476 KB  ->  588,817 Ethiopic characters
                      Dillmann's Ethiopic Octateuch + Ludolf's Psalter,
                      digitized by Ran HaCohen (same scholar as the Enoch
                      text fetched by fetch_ethiopic_corpus.py).

LICENCE: the Ge'ez module is "Copyrighted; Free non-commercial distribution".
Redistribution requires keeping the © Ran HaCohen notice. This script writes
that notice into LICENCE.txt beside the extracted text -- do not delete it,
and do not use this corpus commercially.

Usage:
    python scripts/fetch_sword_bible.py --module Geez --language geez \\
        --output-dir data/corpus_raw/geez_bible
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import requests

MODULE_URL = "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/{module}.zip"
USER_AGENT = "AXUM-Rover-Corpus/1.0 (heritage OCR research)"

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
OSIS_TAG_RE = re.compile(r"<[^>]+>")
# Ethiopic word separator and full stop; keep them, they are real punctuation.
KEEP_PUNCT = "\u1361\u1362"


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if ETHIOPIC_RE.match(c)) / len(dense)


def clean_osis(raw: str, min_ratio: float) -> list[str]:
    """
    Strip OSIS markup and keep lines that are genuinely Ethiopic.

    Args:
        raw: Decompressed block text
        min_ratio: Per-line Ethiopic density threshold

    Returns:
        Cleaned Ethiopic lines
    """
    text = OSIS_TAG_RE.sub("\n", raw)
    lines: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line or not ETHIOPIC_RE.search(line):
            continue
        if ethiopic_ratio(line) < min_ratio:
            continue
        lines.append(line)
    return lines


def read_blocks(archive: zipfile.ZipFile, module: str) -> list[str]:
    """
    Decompress every zText block in a module.

    Args:
        archive: Opened module zip
        module: Module name as used in its DataPath

    Returns:
        One decoded string per block
    """
    names = archive.namelist()
    stems = {
        name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for name in names if name.endswith(".bzz")
    }
    if not stems:
        raise ValueError(f"{module}: no .bzz payload found (not a zText module?)")

    blocks: list[str] = []
    for stem in sorted(stems):
        bzs_name = next(n for n in names if n.endswith(f"{stem}.bzs"))
        bzz_name = next(n for n in names if n.endswith(f"{stem}.bzz"))
        index = archive.read(bzs_name)
        payload = archive.read(bzz_name)

        for record in range(len(index) // 12):
            offset, compressed, _raw_size = struct.unpack_from("<III", index, record * 12)
            chunk = payload[offset:offset + compressed]
            try:
                blocks.append(zlib.decompress(chunk).decode("utf-8", "ignore"))
            except zlib.error as exc:
                print(f"  {stem} block {record}: decompress failed ({exc})")
    return blocks


def read_licence(archive: zipfile.ZipFile) -> str:
    """Pull the About/Copyright block out of the module .conf."""
    for name in archive.namelist():
        if name.endswith(".conf"):
            return archive.read(name).decode("utf-8", "ignore")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="SWORD module name, e.g. Geez")
    parser.add_argument("--language", required=True, help="Label, e.g. geez")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-ethiopic-ratio", type=float, default=0.6)
    parser.add_argument("--lines-per-file", type=int, default=200,
                        help="Chunk size, so downstream tooling sees many files")
    args = parser.parse_args()

    url = MODULE_URL.format(module=args.module)
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

    blocks = read_blocks(archive, args.module)
    lines: list[str] = []
    for block in blocks:
        lines.extend(clean_osis(block, args.min_ethiopic_ratio))

    if not lines:
        print("No Ethiopic lines recovered", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(lines), args.lines_per_file):
        part = lines[start:start + args.lines_per_file]
        path = args.output_dir / f"{args.language}_{args.module.lower()}_{written:04d}.txt"
        path.write_text("\n".join(part), encoding="utf-8")
        written += 1

    licence = read_licence(archive)
    (args.output_dir / "LICENCE.txt").write_text(
        f"Source: CrossWire SWORD module '{args.module}'\n"
        f"URL: {url}\n\n{licence}\n",
        encoding="utf-8",
    )

    total = sum(len(ETHIOPIC_RE.findall(line)) for line in lines)
    print(f"{args.language}: {len(lines):,} lines across {written} files")
    print(f"  Ethiopic characters: {total:,}")
    print(f"  Licence recorded in {args.output_dir / 'LICENCE.txt'} — "
          f"non-commercial use only, keep the attribution")
    print(f"\nNEXT: python scripts/prepare_restoration_corpus.py "
          f"--input-dir {args.output_dir} "
          f"--output data/restoration_corpus_{args.language}_bible.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
