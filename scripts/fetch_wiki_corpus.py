"""
fetch_wiki_corpus.py — collect Ethiopic-script text from Wikimedia sources.

WHAT: builds a plain-text corpus for one language, from either a Wikimedia XML
dump (bulk, preferred) or the MediaWiki API (targeted search). Writes one .txt
per article plus a provenance manifest.

WHY one script for both languages: the Ge'ez-vs-Amharic scaling comparison is
only meaningful if corpus SIZE is the one thing that differs. Two scrapers
would diverge in cleaning, line splitting and filtering, and any accuracy gap
would then be unattributable. Same code path, different --language.

WHY dumps rather than the API for bulk. Measured, not assumed:
  * MediaWiki refuses batched whole-article extracts. It rewrites the request
    and warns: 'exlimit was too large for a whole article extracts request,
    lowered to 1'. So the API costs ONE request per article.
  * At that rate am.wikipedia.org returns HTTP 429 within a few dozen requests.
  * Walking allpages alphabetically yields year-stubs ('1009 እ.ኤ.አ.') first --
    the least useful text in the wiki.
  * The whole amwiki dump is 9.2 MB. One download beats 15,721 throttled calls.

Dump availability, checked 2026-08:
    amwiki    9.2 MB   Amharic, 15,721 articles / 2,113,801 words
    tiwiki    1.3 MB   Tigrinya, 366 articles
    tigwiki   1.3 MB   Tigre, 64 articles
    gezwiki   404      Ge'ez has NO Wikipedia edition and no dump.

For Ge'ez use --source api against wikisource.org with --search, or the
chapter scraper in fetch_ethiopic_corpus.py.

Usage:
    python scripts/fetch_wiki_corpus.py --wiki amwiki --language amharic \\
        --output-dir data/corpus_raw/amharic

    python scripts/fetch_wiki_corpus.py --source api --wiki wikisource.org \\
        --language geez --search "ግዕዝ" --max-articles 500 \\
        --output-dir data/corpus_raw/geez

Then chunk with the existing pipeline:
    python scripts/prepare_restoration_corpus.py \\
        --input-dir data/corpus_raw/amharic \\
        --output data/restoration_corpus_amharic.json
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import requests

# Ethiopic main block plus supplement/extended ranges. Ge'ez uses the core;
# Amharic adds fidels for sounds Ge'ez lacks, but they sit in the same block --
# verified: all 8 Amharic-specific base fidels are already in GEEZ_CHARSET,
# which is why one charset and one architecture can serve both languages.
ETHIOPIC_PATTERN = re.compile(
    r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]"
)

USER_AGENT = "AXUM-Rover-Corpus/1.0 (heritage OCR research)"
DUMP_URL = (
    "https://dumps.wikimedia.org/{wiki}/latest/{wiki}-latest-pages-articles.xml.bz2"
)
API_MAX_ATTEMPTS = 5


@dataclass
class ArticleRecord:
    """Provenance for one collected article."""

    title: str
    pageid: int
    language: str
    source: str
    url: str
    ethiopic_chars: int
    lines: int
    filename: str


# ── Wikitext cleaning ────────────────────────────────────────────

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
FILE_LINK_RE = re.compile(r"\[\[\s*(?:File|Image|ስዕል|ፋይል)\s*:[^\]]*\]\]", re.IGNORECASE)
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://\S+\s+([^\]]*)\]")
BARE_URL_RE = re.compile(r"\[?(?:https?|ftp)://\S+\]?")
HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
BOLD_ITALIC_RE = re.compile(r"'{2,5}")
LIST_PREFIX_RE = re.compile(r"^[\*\#:;]+\s*", re.MULTILINE)


def strip_balanced(text: str, opener: str, closer: str) -> str:
    """
    Remove balanced {{...}} / {|...|} regions, including nested ones.

    Args:
        text: Wikitext
        opener: Opening delimiter
        closer: Closing delimiter

    Returns:
        Text with balanced regions removed

    A regex cannot do this correctly: templates nest, and a non-greedy pattern
    stops at the first closer, leaving orphaned tails that survive cleaning and
    land in the corpus as pseudo-sentences.
    """
    out: list[str] = []
    depth = 0
    index = 0
    length = len(text)
    while index < length:
        if text.startswith(opener, index):
            depth += 1
            index += len(opener)
        elif depth and text.startswith(closer, index):
            depth -= 1
            index += len(closer)
        else:
            if not depth:
                out.append(text[index])
            index += 1
    return "".join(out)


def resolve_wikilinks(text: str) -> str:
    """Reduce [[target|label]] to label and [[target]] to target."""

    def replace(match: re.Match) -> str:
        inner = match.group(1)
        return inner.split("|")[-1] if "|" in inner else inner

    return re.sub(r"\[\[([^\[\]]+)\]\]", replace, text)


def wikitext_to_plain(raw: str) -> str:
    """
    Convert MediaWiki wikitext into rough plain text.

    Args:
        raw: Raw revision body from a dump or the API

    Returns:
        Plain text. Markup residue is tolerated because the Ethiopic-density
        filter downstream discards any line that is mostly non-Ethiopic.
    """
    text = COMMENT_RE.sub(" ", raw)
    text = REF_RE.sub(" ", text)
    text = strip_balanced(text, "{{", "}}")
    text = strip_balanced(text, "{|", "|}")
    text = FILE_LINK_RE.sub(" ", text)
    text = resolve_wikilinks(text)
    text = EXTERNAL_LINK_RE.sub(r"\1", text)
    text = BARE_URL_RE.sub(" ", text)
    text = HEADING_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = BOLD_ITALIC_RE.sub("", text)
    text = LIST_PREFIX_RE.sub("", text)
    return text


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if ETHIOPIC_PATTERN.match(c)) / len(dense)


def clean_lines(text: str, min_ratio: float) -> str:
    """
    Keep only lines that are genuinely Ethiopic prose.

    Args:
        text: Plain text
        min_ratio: Reject lines below this Ethiopic density, 0-1

    Returns:
        Cleaned text, empty when nothing survives

    A per-line density test rather than a bare "contains Ethiopic" check:
    Amharic articles mix Latin names, ISBNs and dates into otherwise-Ethiopic
    lines, and a contains-check admits reference lists that are mostly Latin.
    """
    kept: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line or not ETHIOPIC_PATTERN.search(line):
            continue
        if ethiopic_ratio(line) < min_ratio:
            continue
        kept.append(line)
    return "\n".join(kept)


def slugify(title: str, limit: int = 60) -> str:
    """Filesystem-safe filename stem from an article title."""
    slug = re.sub(r"[^\w\u1200-\u137F]+", "_", title, flags=re.UNICODE).strip("_")
    return slug[:limit] or "untitled"


# ── Dump source ──────────────────────────────────────────────────

def download_dump(wiki: str, cache_dir: Path, session: requests.Session) -> Path:
    """
    Download a wiki's pages-articles dump, reusing any cached copy.

    Args:
        wiki: Dump name, e.g. "amwiki"
        cache_dir: Where to store the archive
        session: Shared HTTP session

    Returns:
        Path to the local dump file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{wiki}-latest-pages-articles.xml.bz2"
    if target.exists() and target.stat().st_size > 0:
        print(f"Using cached dump: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    url = DUMP_URL.format(wiki=wiki)
    print(f"Downloading {url}")
    partial = target.with_suffix(".part")
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"  {written / 1e6:6.1f} / {total / 1e6:.1f} MB", end="\r")
    partial.replace(target)
    print(f"\nSaved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def _child(element, name: str):
    """Find a direct child by local tag name, ignoring XML namespace."""
    for candidate in element:
        if candidate.tag.endswith("}" + name) or candidate.tag == name:
            return candidate
    return None


def iter_dump_pages(dump_path: Path):
    """
    Stream (title, pageid, wikitext) for main-namespace, non-redirect pages.

    Args:
        dump_path: Local .xml.bz2 dump

    Yields:
        Tuples of title, pageid, raw wikitext

    Parsed with iterparse and cleared per element: the decompressed XML is far
    larger than the archive, so building a full tree is not viable even at
    Amharic's modest size.
    """
    with bz2.open(dump_path, "rb") as stream:
        for _event, element in ElementTree.iterparse(stream, events=("end",)):
            if not (element.tag.endswith("}page") or element.tag == "page"):
                continue

            namespace = _child(element, "ns")
            title_el = _child(element, "title")
            revision = _child(element, "revision")
            in_main = namespace is not None and (namespace.text or "").strip() == "0"

            if in_main and _child(element, "redirect") is None and revision is not None:
                text_el = _child(revision, "text")
                id_el = _child(element, "id")
                if text_el is not None and text_el.text and title_el is not None:
                    pageid = int((id_el.text or "-1").strip()) if id_el is not None else -1
                    yield title_el.text or "", pageid, text_el.text

            element.clear()


# ── API source ───────────────────────────────────────────────────

def api_get(session: requests.Session, host: str, params: dict) -> dict:
    """Call a MediaWiki API endpoint, backing off on HTTP 429."""
    payload = {**params, "format": "json", "formatversion": "2"}
    delay = 2.0
    for attempt in range(1, API_MAX_ATTEMPTS + 1):
        response = session.get(f"https://{host}/w/api.php", params=payload, timeout=30)
        if response.status_code == 429:
            if attempt == API_MAX_ATTEMPTS:
                response.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()
    return {}


def iter_api_articles(
    session: requests.Session, host: str, search: str, limit: int, delay: float
):
    """
    Yield (title, pageid, wikitext) for search hits on a live wiki.

    Args:
        session: Shared HTTP session
        host: Wiki hostname, e.g. "wikisource.org"
        search: Search term
        limit: Maximum articles
        delay: Seconds between calls

    Uses revision content rather than extracts: MediaWiki caps whole-article
    extracts at one page per request, but revision content batches normally.
    """
    seen = 0
    cont: dict = {}
    while seen < limit:
        payload = api_get(session, host, {
            "action": "query", "generator": "search", "gsrsearch": search,
            "gsrnamespace": 0, "gsrlimit": 20,
            "prop": "revisions", "rvprop": "content", "rvslots": "main",
            **cont,
        })
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return

        for page in pages:
            if seen >= limit:
                return
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            content = revisions[0].get("slots", {}).get("main", {}).get("content")
            if not content:
                continue
            yield page.get("title", ""), page.get("pageid", -1), content
            seen += 1

        if "continue" not in payload:
            return
        cont = payload["continue"]
        time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", required=True,
                        help="Dump name (amwiki) or API host (wikisource.org)")
    parser.add_argument("--language", required=True,
                        help="Manifest label, e.g. amharic or geez")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("dump", "api"), default="dump")
    parser.add_argument("--search", default=None,
                        help="Search term, required for --source api")
    parser.add_argument("--max-articles", type=int, default=0,
                        help="0 means no limit (dump mode)")
    parser.add_argument("--min-chars", type=int, default=80,
                        help="Skip articles with fewer Ethiopic characters")
    parser.add_argument("--min-ethiopic-ratio", type=float, default=0.6,
                        help="Per-line Ethiopic density threshold, 0-1")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/dumps"))
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.source == "api" and not args.search:
        parser.error("--source api requires --search")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        if args.source == "dump":
            supply = iter_dump_pages(
                download_dump(args.wiki, args.cache_dir, session)
            )
        else:
            supply = iter_api_articles(
                session, args.wiki, args.search,
                args.max_articles or 500, args.delay,
            )
    except requests.RequestException as exc:
        print(f"Could not reach {args.wiki}: {exc}", file=sys.stderr)
        return 1

    records: list[ArticleRecord] = []
    skipped = 0
    scanned = 0

    for title, pageid, raw in supply:
        scanned += 1
        text = clean_lines(wikitext_to_plain(raw), args.min_ethiopic_ratio)
        ethiopic_count = sum(1 for c in text if ETHIOPIC_PATTERN.match(c))

        if ethiopic_count < args.min_chars:
            skipped += 1
        else:
            filename = f"{args.language}_{slugify(title)}_{pageid}.txt"
            (args.output_dir / filename).write_text(text, encoding="utf-8")
            records.append(ArticleRecord(
                title=title, pageid=pageid, language=args.language,
                source=f"{args.source}:{args.wiki}",
                url=f"https://{args.wiki}/?curid={pageid}",
                ethiopic_chars=ethiopic_count,
                lines=len(text.splitlines()), filename=filename,
            ))

        if scanned % 250 == 0:
            print(f"  scanned {scanned} | kept {len(records)} | "
                  f"skipped {skipped}", end="\r")
        if args.max_articles and len(records) >= args.max_articles:
            break

    total_chars = sum(r.ethiopic_chars for r in records)
    manifest = {
        "language": args.language,
        "wiki": args.wiki,
        "source": args.source,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "articles_scanned": scanned,
        "articles_kept": len(records),
        "articles_skipped": skipped,
        "total_ethiopic_chars": total_chars,
        "licence": "CC BY-SA 4.0 (Wikimedia) — attribute per-article URL",
        "filters": {
            "min_chars": args.min_chars,
            "min_ethiopic_ratio": args.min_ethiopic_ratio,
        },
        "articles": [asdict(r) for r in records],
    }
    manifest_path = args.output_dir / f"{args.language}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{args.language}: scanned {scanned}, kept {len(records)}, "
          f"skipped {skipped}")
    print(f"  Ethiopic characters: {total_chars:,}")
    print(f"  Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
