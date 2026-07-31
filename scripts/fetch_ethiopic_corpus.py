"""
fetch_ethiopic_corpus.py — downloads chapter pages from a HaCohen-style
Ethiopic-text site (one HTML page per chapter) and saves clean plain-text
.txt files, one per chapter, ready for scripts/prepare_restoration_corpus.py.

WHAT: for a given base URL pattern and chapter range, fetches each page,
strips the leading verse-reference link and surrounding HTML/markdown
noise, and writes the remaining Ge'ez text as a plain .txt file.

WHY a script instead of manual save-as: consistent, repeatable, produces
uniformly-named files with no per-page manual steps — and importantly,
this is YOUR machine doing the fetching directly against the source site,
not text being relayed and reproduced through a third party.

Usage example (Book of Enoch, Dillmann edition, chapters 2-22):
    python fetch_ethiopic_corpus.py \\
        --base-url "https://www.tau.ac.il/~hacohen/Henoch/Henoch" \\
        --start 2 --end 22 \\
        --source-name enoch_dillmann \\
        --output-dir data/restoration_corpus_raw

Respects the source server: adds a short delay between requests rather
than hammering it with 20+ rapid-fire calls.
"""

import argparse
import html
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

# Strips a leading verse-reference link, e.g.:
#   "[3](http://enoksbok.se/cgi-bin/slaupp.cgi?...) ጠየቁ ፡ ..."
# down to just the Ge'ez text that follows it.
LEADING_REF_PATTERN = re.compile(r"^\[\d+\]\([^)]*\)\s*")

# Strips a leading verse/chapter number and the run of spaces/figure-spaces that
# follows it, e.g. "2\u2007\u2007\u2007\u2007ጠየቁ..." -> "ጠየቁ...". This is separate
# from LEADING_REF_PATTERN above (that one handles the markdown-link style marker;
# this handles a bare numeral prefix with no link).
LEADING_VERSE_NUM_PATTERN = re.compile(r"^[0-9፩-፼]+[\s\u2007]+")

# Any real body line must contain actual Ethiopic script. A line with none is a
# heading/title/navigation artifact (e.g. a bare "Henoch 6" line), not text —
# confirmed necessary after finding "Henoch N" leaking into every chapter's first
# chunk on real fetched files: the title-stripping below assumed markdown
# frontmatter ("title: ...") but the real saved pages show it as a bare heading
# line with no such prefix, so this catches it regardless of book name.
ETHIOPIC_CHAR_PATTERN = re.compile(r"[\u1200-\u137F]")

# Strips basic HTML tags if the raw page HTML is fetched instead of
# pre-rendered text (safety net — harmless if there are no tags to strip).
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def clean_page_text(raw: str) -> str:
    """WHAT: strips markup/reference-link noise, keeps the Ge'ez body text.
    WHY: the source page includes a clickable verse-reference marker before
    the actual text — that's a navigation artifact, not part of the source
    text, and would corrupt chunk boundaries if left in."""
    text = HTML_TAG_PATTERN.sub("", raw)
    text = html.unescape(text)  # decode &#4808;-style numeric entities into real Unicode
                                 # characters — this site encodes Ge'ez this way in its
                                 # page source; without this step every character comes
                                 # through as literal "&#NNNN;" text, not the letter itself.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Drop any line that's just a title/frontmatter artifact (e.g. "title: ...", "---")
    lines = [l for l in lines if l not in ("---",) and not l.lower().startswith("title:")]
    # Drop any line with zero Ethiopic characters — catches bare heading lines like
    # "Henoch 6" that survive the check above (confirmed necessary on real fetched files).
    lines = [l for l in lines if ETHIOPIC_CHAR_PATTERN.search(l)]
    cleaned_lines = [LEADING_REF_PATTERN.sub("", l) for l in lines]
    cleaned_lines = [LEADING_VERSE_NUM_PATTERN.sub("", l) for l in cleaned_lines]
    return "\n".join(l for l in cleaned_lines if l)


def fetch_chapter(base_url: str, chapter_num: int, delay_s: float) -> str:
    """WHAT: fetches one chapter page and returns cleaned text.
    WHY: isolated so the main loop stays simple and each failure is
    reported per-chapter rather than aborting the whole batch."""
    url = f"{base_url}%20{chapter_num}.html"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"  # defensive — not the actual bug (see html.unescape above),
                             # but no reason to trust requests' guessed encoding either
    time.sleep(delay_s)
    return clean_page_text(resp.text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                         help="URL up to and excluding '%%20<N>.html', e.g. "
                              "'https://www.tau.ac.il/~hacohen/Henoch/Henoch'")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--source-name", required=True,
                         help="Used as the output filename prefix and the "
                              "'source' label carried into every chunk record.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to wait between requests (be polite to the source server).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for n in range(args.start, args.end + 1):
        try:
            text = fetch_chapter(args.base_url, n, args.delay)
            if not text.strip():
                print(f"  chapter {n}: fetched but empty after cleaning — skipping, check manually")
                failed.append(n)
                continue
            out_path = args.output_dir / f"{args.source_name}_ch{n:03d}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"  chapter {n}: saved -> {out_path.name} ({len(text)} chars)")
            ok += 1
        except requests.exceptions.RequestException as e:
            print(f"  chapter {n}: FAILED — {e}")
            failed.append(n)

    print(f"\nDone: {ok} saved, {len(failed)} failed.")
    if failed:
        print(f"Failed chapters: {failed} — check these manually, page numbering may not be continuous.")
    print(f"\nNEXT STEP: python scripts/prepare_restoration_corpus.py "
          f"--input-dir {args.output_dir} --output data/restoration_corpus_chunked.json")


if __name__ == "__main__":
    main()