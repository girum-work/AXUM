"""
prepare_restoration_corpus.py — one-time preprocessing: raw corpus texts ->
phrase/formula-level chunks, ready for corpus_retriever.py to index.

WHAT: reads raw .txt source files (Kebra Nagast, Ge'ez Book of Enoch, known
Aksumite royal inscriptions — one file per source, plain UTF-8 text) and
splits each into phrase/formula-level chunks, writing a single JSON file:
a list of {"id", "text", "source"} records.

WHY phrase-level, not document-level: retrieval needs to match a damaged
FRAGMENT against the closest known FORMULA, not against an entire text.
Document-level chunks would retrieve too much irrelevant surrounding context.

*** REVIEW REQUIRED ***
Chunk-boundary quality is the single most likely silent failure mode in the
whole restoration pipeline — bad boundaries quietly produce bad retrieval
with no error. The heuristic split below (Ge'ez word-divider/punctuation-based,
falling back to line breaks) is a reasonable starting point, NOT a substitute
for review. Heritage & Conservation Specialist should review a sample of the
output chunks before this corpus is trusted for real retrieval.

Usage:
    python scripts/prepare_restoration_corpus.py \\
        --input-dir data/restoration_corpus_raw \\
        --output data/restoration_corpus_chunked.json
"""

import argparse
import json
import re
from pathlib import Path

from loguru import logger

# Ge'ez punctuation commonly used as phrase/clause boundaries.
# ':' (word divider, U+1361) is too fine-grained alone (splits every word) —
# used only as a fallback secondary split within an over-long primary chunk.
# '።' (full stop, U+1362) and '፤'/'፣' (colon/comma, U+1364/U+1363) are the
# primary phrase-boundary candidates.
PRIMARY_BOUNDARY_CHARS = "።፤፣፥፦"
SECONDARY_BOUNDARY_CHAR = "፡"

MIN_CHUNK_CHARS = 8    # discard fragments too short to be a useful retrieval target
MAX_CHUNK_CHARS = 120  # split further if a "sentence" is implausibly long (likely mis-split source)


def split_into_chunks(raw_text: str) -> list:
    """WHAT: splits raw text into phrase-level strings.
    WHY: primary split on sentence-level punctuation; long results get a
    secondary split on the word-divider so no single chunk is unreasonably
    long (keeps chunks close to "one formula" in scale)."""
    # Primary split — keep the boundary character attached to the preceding chunk
    pattern = f"([{re.escape(PRIMARY_BOUNDARY_CHARS)}])"
    pieces = re.split(pattern, raw_text)
    primary_chunks = []
    buf = ""
    for piece in pieces:
        buf += piece
        if piece in PRIMARY_BOUNDARY_CHARS:
            primary_chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        primary_chunks.append(buf.strip())

    # Fall back to line breaks if punctuation-based split produced ~nothing
    # (e.g. source text has no primary boundary punctuation at all)
    if len(primary_chunks) <= 1:
        primary_chunks = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Secondary split for implausibly long "sentences" — greedily group words up
    # to MAX_CHUNK_CHARS rather than splitting on every single word-divider.
    # WHY: splitting on every divider degrades to near-word-level fragments, most
    # of which then fall under MIN_CHUNK_CHARS and get silently discarded by the
    # filter below — confirmed on real text, this destroyed 18 of 20 words from
    # one legitimate long sentence with no error. Greedy grouping never emits a
    # fragment smaller than a full accumulated word-group.
    final_chunks = []
    for chunk in primary_chunks:
        if len(chunk) <= MAX_CHUNK_CHARS:
            final_chunks.append(chunk)
        else:
            words = [w for w in chunk.split(SECONDARY_BOUNDARY_CHAR) if w.strip()]
            buf = ""
            for w in words:
                candidate = (buf + SECONDARY_BOUNDARY_CHAR + w) if buf else w
                if len(candidate) > MAX_CHUNK_CHARS and buf:
                    # Keep the divider that originally sat between buf's last word
                    # and w — matches real Ge'ez convention (word-divider trails
                    # each word) and closes a one-character loss confirmed via
                    # reconstruction-check against real fetched chapters.
                    final_chunks.append((buf + SECONDARY_BOUNDARY_CHAR).strip())
                    buf = w
                else:
                    buf = candidate
            if buf.strip():
                final_chunks.append(buf.strip())

    # Never silently discard content. Short fragments (common for legitimate
    # ፤-separated list-style clauses) get merged into a neighboring chunk rather
    # than dropped — the previous MIN_CHUNK_CHARS filter was discarding real
    # words/final sentences outright, confirmed via reconstruction-check against
    # real chapters (one case cut off a chapter's final sentence entirely).
    result = [c for c in final_chunks if c]
    merged = []
    i = 0
    while i < len(result):
        chunk = result[i]
        if len(chunk) < MIN_CHUNK_CHARS:
            if merged:
                merged[-1] = (merged[-1] + " " + chunk).strip()
            elif i + 1 < len(result):
                result[i + 1] = (chunk + " " + result[i + 1]).strip()
            else:
                merged.append(chunk)  # entire chapter is one short fragment — keep it
        else:
            merged.append(chunk)
        i += 1
    return merged


def prepare_corpus(input_dir: Path, output_path: Path) -> None:
    """WHAT: processes every .txt file in input_dir into the final chunked JSON.
    WHY: single entry point, one file per known source, source name preserved
    per-chunk for traceability in restoration output."""
    records = []
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        logger.error(f"No .txt files found in {input_dir} — nothing to prepare.")
        return

    for txt_file in txt_files:
        source_name = txt_file.stem
        raw_text = txt_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(raw_text)
        for i, chunk_text in enumerate(chunks):
            records.append({
                "id": f"{source_name}_{i:04d}",
                "text": chunk_text,
                "source": source_name,
            })
        logger.info(f"{txt_file.name}: {len(chunks)} chunks")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(records)} total chunks to {output_path}")
    logger.warning(
        "REVIEW REQUIRED: Heritage & Conservation Specialist should sample-check "
        "chunk boundaries in the output before this corpus is used for real retrieval."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                         help="Directory of raw .txt source files (one per known text)")
    parser.add_argument("--output", type=Path, required=True,
                         help="Output path for chunked JSON corpus")
    args = parser.parse_args()
    prepare_corpus(args.input_dir, args.output)