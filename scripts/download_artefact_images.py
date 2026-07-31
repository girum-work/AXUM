"""
AXUM ROVER — Artefact Classification Image Downloader
======================================================
Populates data/artefact_classes/ with open-licence museum images for
MobileNetV2 / YOLO11 artefact classifier training (turntable top-down).

Approved APIs (priority order):
  1. Wikimedia Commons (CC BY, CC BY-SA, CC0, public domain)
  2. The Metropolitan Museum of Art Open Access (CC0)

Run from project root (venv activated):
  python scripts/download_artefact_images.py
  python scripts/download_artefact_images.py --ideal
  python scripts/download_artefact_images.py --class pottery --dry-run
  python scripts/download_artefact_images.py --report-only

Writes:
  data/artefact_classes/{class}/{source}_{class}_{NNN}.jpg
  data/artefact_classes/metadata.csv
  data/artefact_classes/SOURCES.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from loguru import logger
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ARTEFACT_CLASS_IDEAL_COUNTS,
    ARTEFACT_CLASS_MIN_COUNTS,
    ARTEFACT_CLASSES_DIR,
    ARTEFACT_DATASET_IDEAL_TOTAL,
    ARTEFACT_DATASET_MIN_TOTAL,
    ARTEFACT_MAX_CLASS_RATIO,
    ARTEFACT_MAX_NEAR_DUPES_PER_SOURCE,
    ARTEFACT_METADATA_CSV,
    ARTEFACT_MIN_COVERAGE_RATIO,
    ARTEFACT_MIN_IMAGE_PX,
    ARTEFACT_MIN_SHARPNESS_VAR,
    ARTEFACT_NEAR_DUPE_HAMMING,
    BM_REQUEST_TIMEOUT,
    BM_SEARCH_API,
    BM_SPARQL_API,
    COIN_SUBCLASS_MIN_IMAGES,
    COIN_SUBCLASS_TARGETS,
    COIN_SUBTYPES_DIR,
    OBJ_CLASSES,
    SMITHSONIAN_API_KEY,
)
from src.object_detection.artefact_label_rules import (
    COIN_POSITIVE,
    FOREIGN_COIN_REJECT,
    is_ethiopia_related,
)
from src.object_detection.artefact_label_rules import (
    LabelAction,
    build_met_context,
    decide_label,
    filename_to_title_hint,
)

USER_AGENT = (
    "AXUM-Rover/1.0 (WRO 2026 educational robot; "
    "contact: axum-rover-student-project; local non-commercial dataset build)"
)
REQUEST_TIMEOUT = 45
REQUEST_DELAY_S = 0.8
WIKIMEDIA_IMAGE_DELAY_S = 2.5
WIKIMEDIA_RETRY_AFTER_S = 30
MAX_HTTP_RETRIES = 4
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
MET_SEARCH_API = (
    "https://collectionapi.metmuseum.org/public/collection/v1/search"
)
MET_OBJECT_API = (
    "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
)
SI_SEARCH_API = "https://api.si.edu/openaccess/api/v1.0/search"
SI_CONTENT_API = "https://api.si.edu/openaccess/api/v1.0/content/{id}"
WIKIMEDIA_NEG_KEYWORDS = (
    "-Olympics -ceremony -diagram -chart -map -logo -icon -flag -poster "
    "-painting -canvas -portrait -Adoration -Annunciation -fresco "
    "-engraving -illustration -tapestry"
)

METADATA_FIELDS = [
    "filename",
    "class",
    "source_url",
    "license",
    "width_px",
    "height_px",
    "notes",
]

# Approved licence substrings (case-insensitive); reject NC / all-rights / FAL-only.
LICENSE_ALLOW = (
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
    "public domain",
    "pd-art",
    "pd-us",
    "no restrictions",
    "creative commons attribution",
)
LICENSE_DENY = (
    "nc ",
    "nc-",
    "-nc",
    "non-commercial",
    "all rights reserved",
    "copyrighted",
    "fair use",
    "fal",
    "cc by-nd",
    "cc-by-nd",
    "no derivatives",
)

# Ethiopian cultural heritage — all periods (ancient through modern historical).
# AXUM project name ≠ Aksum-only scope.
CLASS_SEARCH_QUERIES: dict[str, list[tuple[str, str]]] = {
    "pottery": [
        ("britishmuseum", "Ethiopia pottery"),
        ("britishmuseum", "Ethiopia ceramic"),
        ("smithsonian", "Ethiopia pottery"),
        ("met", "Ethiopia"),
        ("met", "Ethiopia pottery"),
        ("met", "Aksum ceramic"),
        ("wikimedia", "Aksumite pottery"),
        ("wikimedia", "Ethiopia ceramic archaeological"),
        ("wikimedia", "Ethiopian terracotta"),
        ("wikimedia", "Lalibela pottery"),
        ("wikimedia", "Tigray pottery Ethiopia"),
        ("wikimedia", "Gondar pottery Ethiopia"),
        ("wikimedia", "Ethiopian clay pot museum"),
        ("wikimedia", "Horn of Africa pottery"),
        ("met", "Ethiopia pottery"),
        ("met", "Aksum ceramic"),
        ("smithsonian", "Ethiopia pottery"),
    ],
    "stone_carving": [
        ("britishmuseum", "Ethiopia stela"),
        ("britishmuseum", "Aksum obelisk"),
        ("smithsonian", "Ethiopia stela"),
        ("met", "Ethiopia"),
        ("met", "Aksum stela"),
        ("met", "Ethiopia sculpture"),
        ("wikimedia", "Aksum stelae"),
        ("wikimedia", "Ethiopian stela"),
        ("wikimedia", "Axum obelisk"),
        ("wikimedia", "Lalibela rock church"),
        ("wikimedia", "Tigray rock hewn church"),
        ("wikimedia", "Gondar castle stone"),
        ("wikimedia", "Ethiopian stone carving"),
        ("wikimedia", "Harar stone"),
        ("wikimedia", "Ethiopian stone relief"),
        ("wikimedia", "Hawzen stela"),
        ("met", "Ethiopia sculpture"),
        ("met", "Aksum stela"),
        ("smithsonian", "Ethiopia stela"),
    ],
    "coin": [
        ("britishmuseum", "Ethiopia coin"),
        ("britishmuseum", "Aksumite coin"),
        ("smithsonian", "Ethiopia coin"),
        ("met", "Ethiopia"),
        ("met", "Ethiopia coin"),
        ("met", "Aksum coin"),
        ("wikimedia", "Aksumite coin"),
        ("wikimedia", "King Ezana coin"),
        ("wikimedia", "Ethiopian coin"),
        ("wikimedia", "Menelik coin"),
        ("wikimedia", "Haile Selassie coin"),
        ("wikimedia", "Ethiopia birr coin"),
        ("wikimedia", "Ethiopian 1 birr"),
        ("wikimedia", "Ethiopian cent coin"),
        ("wikimedia", "Ethiopia numismatics"),
        ("wikimedia", "Axum gold coin"),
        ("met", "Ethiopia coin"),
        ("met", "Aksum coin"),
        ("smithsonian", "Ethiopia coin"),
        ("smithsonian", "Menelik Ethiopia"),
    ],
    "inscription_fragment": [
        ("britishmuseum", "Ethiopia inscription"),
        ("britishmuseum", "Ge'ez inscription"),
        ("smithsonian", "Ethiopia inscription"),
        ("met", "Ethiopia"),
        ("met", "Ethiopia inscription"),
        ("met", "Ge'ez Ethiopia"),
        ("wikimedia", "Ge'ez inscription"),
        ("wikimedia", "Ethiopic inscription"),
        ("wikimedia", "Axum inscription"),
        ("wikimedia", "Ezana stone inscription"),
        ("wikimedia", "Hawzen inscription"),
        ("wikimedia", "Tigray inscription stone"),
        ("wikimedia", "Lalibela inscription"),
        ("wikimedia", "Ethiopian epigraphy"),
        ("wikimedia", "South Arabian inscription Ethiopia"),
        ("wikimedia", "Ethiopian ostracon"),
        ("met", "Ethiopia inscription"),
        ("met", "Ge'ez Ethiopia"),
        ("smithsonian", "Ethiopia inscription"),
    ],
    "other": [
        ("britishmuseum", "Ethiopia cross"),
        ("britishmuseum", "Ethiopia antiquities"),
        ("smithsonian", "Ethiopia artifact"),
        ("met", "Ethiopia"),
        ("met", "Ethiopia cross"),
        ("met", "Ethiopia metalwork"),
        ("wikimedia", "Ethiopian cross pendant"),
        ("wikimedia", "Ethiopian processional cross"),
        ("wikimedia", "Ethiopian antiquities"),
        ("wikimedia", "National Museum of Ethiopia"),
        ("wikimedia", "Ethiopian bronze"),
        ("wikimedia", "Ethiopian jewelry"),
        ("wikimedia", "Ethiopian metalwork"),
        ("wikimedia", "Ethiopian religious artifact"),
        ("wikimedia", "Lalibela cross"),
        ("met", "Ethiopia cross"),
        ("met", "Ethiopia metalwork"),
        ("smithsonian", "Ethiopia artifact"),
    ],
}

SOURCE_ABBR = {
    "wikimedia": "wiki",
    "met": "met",
    "smithsonian": "si",
    "britishmuseum": "bm",
}

# User-specified museum search terms (BM → SI → Met priority)
MUSEUM_BASE_QUERIES = {
    "britishmuseum": "Ethiopia",
    "smithsonian": "ethiopic artefact",
    "met": "aksumite",
}


@dataclass
class ImageMeta:
    """One row for metadata.csv."""

    filename: str
    class_name: str
    source_url: str
    license: str
    width_px: int
    height_px: int
    notes: str


@dataclass
class ValidationResult:
    """Outcome of quality checks on raw image bytes."""

    ok: bool
    width: int = 0
    height: int = 0
    coverage: float = 0.0
    sharpness: float = 0.0
    ahash: str = ""
    reason: str = ""


def _label_gate(text: str, target_class: str) -> tuple[str | None, str]:
    """
    Apply global Ethiopian artefact rules before saving an image.

    Args:
        text: Title + filename + API metadata combined
        target_class: Class folder we are filling

    Returns:
        (save_class or None if reject, reason string)
    """
    decision = decide_label(text, target_class, require_ethiopia=True)
    if decision.action == LabelAction.REJECT:
        return None, decision.reason
    if decision.action == LabelAction.MOVE and decision.target_class:
        return decision.target_class, decision.reason
    return target_class, decision.reason


def _session() -> requests.Session:
    """HTTP session with Wikimedia-compliant User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    label: str = "",
) -> requests.Response | None:
    """
    GET with backoff on HTTP 429 (Wikimedia rate limit).

    Waits WIKIMEDIA_RETRY_AFTER_S between retries; returns None after exhaustion.
    """
    for attempt in range(MAX_HTTP_RETRIES):
        try:
            r = session.get(
                url, params=params, timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 429:
                wait = WIKIMEDIA_RETRY_AFTER_S * (attempt + 1)
                logger.warning(
                    f"Rate limited ({label or url[:60]}), "
                    f"sleep {wait}s (attempt {attempt + 1})",
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 429:
                continue
            logger.debug(f"HTTP error {label}: {exc}")
            return None
        except Exception as exc:
            logger.debug(f"Request error {label}: {exc}")
            return None
    return None


def _license_ok(license_str: str) -> bool:
    """True if licence text matches approved open licences."""
    low = (license_str or "").lower().strip()
    if not low:
        return False
    for deny in LICENSE_DENY:
        if deny in low:
            return False
    return any(allow in low for allow in LICENSE_ALLOW)


def _average_hash(img: Image.Image, size: int = 8) -> str:
    """8×8 average hash for near-duplicate detection."""
    g = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p >= avg else "0" for p in pixels)


def _hamming(a: str, b: str) -> int:
    """Bit difference between two average hashes."""
    return sum(x != y for x, y in zip(a, b))


def _content_coverage(gray: np.ndarray) -> float:
    """
    Fraction of pixels likely belonging to the artefact (not flat background).

    Uses deviation from border-median intensity — works for museum photos on
    neutral backgrounds and many turntable-style isolations.
    """
    h, w = gray.shape
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1],
    ])
    ref = float(np.median(border))
    diff = np.abs(gray.astype(np.float32) - ref)
    thresh = max(18.0, 0.12 * (gray.max() - gray.min() + 1))
    return float((diff > thresh).sum()) / diff.size


def _sharpness_variance(gray: np.ndarray) -> float:
    """Laplacian variance — low values indicate blur."""
    edges = np.array(
        Image.fromarray(gray).filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    )
    return float(edges.var())


def validate_image_bytes(data: bytes) -> ValidationResult:
    """
    Reject downloads that fail size, format, sharpness, or coverage rules.

    Requires min ARTEFACT_MIN_IMAGE_PX on both sides, JPEG/PNG decode, and
    heuristics for focus and artefact occupying ≥40% of frame.
    """
    try:
        img = Image.open(BytesIO(data))
        fmt = (img.format or "").upper()
        if fmt not in ("JPEG", "PNG", "JPG"):
            return ValidationResult(False, reason=f"format:{fmt}")
        img = img.convert("RGB")
        w, h = img.size
        if w < ARTEFACT_MIN_IMAGE_PX or h < ARTEFACT_MIN_IMAGE_PX:
            return ValidationResult(
                False, width=w, height=h, reason="too_small",
            )
        gray = np.array(img.convert("L"))
        cov = _content_coverage(gray)
        sharp = _sharpness_variance(gray)
        ah = _average_hash(img)
        if cov < ARTEFACT_MIN_COVERAGE_RATIO:
            return ValidationResult(
                False, width=w, height=h, coverage=cov,
                sharpness=sharp, ahash=ah, reason="low_coverage",
            )
        if sharp < ARTEFACT_MIN_SHARPNESS_VAR:
            return ValidationResult(
                False, width=w, height=h, coverage=cov,
                sharpness=sharp, ahash=ah, reason="blurry",
            )
        return ValidationResult(
            True, width=w, height=h, coverage=cov,
            sharpness=sharp, ahash=ah,
        )
    except Exception as exc:
        return ValidationResult(False, reason=str(exc))


def _count_images(class_dir: Path) -> int:
    """Count jpg/jpeg/png in a class folder."""
    if not class_dir.exists():
        return 0
    n = 0
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            n += 1
    return n


def _next_file_index(class_dir: Path, source_abbr: str, class_name: str) -> int:
    """Next NNN for {source}_{class}_{NNN}.jpg naming."""
    pattern = re.compile(
        rf"^{re.escape(source_abbr)}_{re.escape(class_name)}_(\d+)\.jpg$",
        re.IGNORECASE,
    )
    max_n = 0
    for p in class_dir.glob("*.jpg"):
        m = pattern.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _load_metadata() -> dict[str, ImageMeta]:
    """Load metadata.csv keyed by filename."""
    out: dict[str, ImageMeta] = {}
    if not ARTEFACT_METADATA_CSV.exists():
        return out
    with ARTEFACT_METADATA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename", "").strip()
            if fn:
                out[fn] = ImageMeta(
                    filename=fn,
                    class_name=row.get("class", ""),
                    source_url=row.get("source_url", ""),
                    license=row.get("license", ""),
                    width_px=int(row.get("width_px") or 0),
                    height_px=int(row.get("height_px") or 0),
                    notes=row.get("notes", ""),
                )
    return out


def backfill_metadata_from_disk(metadata: dict[str, ImageMeta]) -> int:
    """
    Add metadata rows for on-disk images missing from metadata.csv.

    Legacy downloads (wiki_File_*.jpg) get placeholder provenance so row
    count matches file count for Section 6 reporting.
    """
    added = 0
    for cls in OBJ_CLASSES:
        class_dir = ARTEFACT_CLASSES_DIR / cls
        if not class_dir.exists():
            continue
        for p in class_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            fn = p.name
            if fn in metadata:
                continue
            try:
                img = Image.open(p)
                w, h = img.size
            except Exception:
                w, h = 0, 0
            notes = "legacy on-disk; re-download for full provenance"
            if fn.startswith("wiki_"):
                src = "legacy_wikimedia"
            elif fn.startswith("met_"):
                src = "legacy_met"
            else:
                src = "manual"
            metadata[fn] = ImageMeta(
                filename=fn,
                class_name=cls,
                source_url="",
                license="unknown (pre-metadata)",
                width_px=w,
                height_px=h,
                notes=notes if src == "manual" else f"{src}; {notes}",
            )
            added += 1
    if added:
        logger.info(f"Backfilled {added} metadata rows from existing files")
    return added


def _save_metadata(all_rows: dict[str, ImageMeta]) -> None:
    """Write metadata.csv (sorted by class, filename)."""
    ARTEFACT_CLASSES_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        all_rows.values(),
        key=lambda r: (r.class_name, r.filename),
    )
    with ARTEFACT_METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "filename": r.filename,
                "class": r.class_name,
                "source_url": r.source_url,
                "license": r.license,
                "width_px": r.width_px,
                "height_px": r.height_px,
                "notes": r.notes,
            })


class ClassTracker:
    """
    Per-class download state: source-id dupes, perceptual near-dupes.

    Limits ARTEFACT_MAX_NEAR_DUPES_PER_SOURCE images per Wikimedia title /
    Met object ID and rejects hashes within Hamming distance threshold.
    """

    def __init__(self) -> None:
        self.source_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int),
        )
        self.hashes: dict[str, list[str]] = defaultdict(list)

    def can_add(
        self, class_name: str, source_id: str, ahash: str,
    ) -> tuple[bool, str]:
        """Return (allowed, rejection_reason)."""
        sc = self.source_counts[class_name]
        if sc[source_id] >= ARTEFACT_MAX_NEAR_DUPES_PER_SOURCE:
            return False, "source_cap"
        for existing in self.hashes[class_name]:
            if _hamming(existing, ahash) <= ARTEFACT_NEAR_DUPE_HAMMING:
                return False, "near_duplicate"
        return True, ""

    def register(self, class_name: str, source_id: str, ahash: str) -> None:
        """Record accepted image for duplicate tracking."""
        self.source_counts[class_name][source_id] += 1
        self.hashes[class_name].append(ahash)


def _save_jpeg(class_dir: Path, data: bytes, filename: str) -> Path | None:
    """Normalize to JPEG on disk."""
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        out = class_dir / filename
        img.save(out, format="JPEG", quality=92)
        return out
    except Exception as exc:
        logger.warning(f"Save failed {filename}: {exc}")
        return None


def fetch_wikimedia(
    session: requests.Session,
    class_name: str,
    class_dir: Path,
    query: str,
    need: int,
    metadata: dict[str, ImageMeta],
    tracker: ClassTracker,
    seen_titles: set[str],
) -> int:
    """
    Search Wikimedia Commons File namespace; download up to `need` images.

    Filters licences, validates quality, writes wiki_{class}_{NNN}.jpg.
    """
    downloaded = 0
    offset = 0
    source_abbr = SOURCE_ABBR["wikimedia"]

    while downloaded < need:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query} {WIKIMEDIA_NEG_KEYWORDS}",
            "gsrnamespace": 6,
            "gsrlimit": min(50, need - downloaded + 15),
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 1024,
        }
        r = _get_with_retry(
            session, WIKIMEDIA_API, params=params, label=f"wiki-search:{query}",
        )
        if r is None:
            logger.warning(f"Wikimedia search failed ({query})")
            break
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break

        for page in pages.values():
            if downloaded >= need:
                break
            title = page.get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            ii = (page.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue

            meta = ii.get("extmetadata") or {}
            license_short = (
                meta.get("LicenseShortName", {}).get("value")
                or meta.get("UsageTerms", {}).get("value")
                or ""
            )
            if not _license_ok(license_short):
                logger.debug(f"Skip licence: {title} ({license_short})")
                continue

            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue

            time.sleep(WIKIMEDIA_IMAGE_DELAY_S)
            img_r = _get_with_retry(
                session, url, label=f"wiki-img:{title[:40]}",
            )
            if img_r is None:
                continue
            raw = img_r.content

            vr = validate_image_bytes(raw)
            if not vr.ok:
                logger.debug(f"Reject {title}: {vr.reason}")
                continue

            label_text = f"{title} {filename_to_title_hint(title)}"
            save_class, gate_reason = _label_gate(label_text, class_name)
            if save_class is None:
                logger.debug(f"Label reject {title}: {gate_reason}")
                continue

            ok, why = tracker.can_add(save_class, title, vr.ahash)
            if not ok:
                logger.debug(f"Dup skip {title}: {why}")
                continue

            save_dir = ARTEFACT_CLASSES_DIR / save_class
            save_dir.mkdir(parents=True, exist_ok=True)
            idx = _next_file_index(save_dir, source_abbr, save_class)
            fname = f"{source_abbr}_{save_class}_{idx:03d}.jpg"
            if _save_jpeg(save_dir, raw, fname) is None:
                continue

            tracker.register(save_class, title, vr.ahash)
            page_url = (
                f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
            )
            note = f"wiki query={query!r}; cov={vr.coverage:.2f}"
            if gate_reason and gate_reason != "ok":
                note += f"; gate={gate_reason}"
            metadata[fname] = ImageMeta(
                filename=fname,
                class_name=save_class,
                source_url=page_url,
                license=license_short,
                width_px=vr.width,
                height_px=vr.height,
                notes=note,
            )
            if save_class == class_name:
                downloaded += 1
                logger.info(
                    f"  [{class_name}] wiki +1 ({downloaded}/{need}) {fname}",
                )
            else:
                logger.info(
                    f"  [{class_name}] wiki reroute → {save_class}: {fname}",
                )

        cont = data.get("continue", {}).get("gsroffset")
        if cont is None:
            break
        offset = cont
        time.sleep(REQUEST_DELAY_S)

    return downloaded


def _bm_sparql_search(
    session: requests.Session,
    query_text: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    """
    Search British Museum SPARQL endpoint for Ethiopian objects with images.

    Returns list of dicts: {id, label, image_url, object_url}.
    Gracefully returns [] if the endpoint is unreachable.
    """
    safe = query_text.replace('"', '\\"')
    sparql = f"""
PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
SELECT DISTINCT ?object ?label ?image WHERE {{
  ?object crm:P1_is_identified_by ?app .
  ?app crm:P190_has_symbolic_content ?label .
  FILTER(
    CONTAINS(LCASE(STR(?label)), "ethiopia") ||
    CONTAINS(LCASE(STR(?label)), "aksum") ||
    CONTAINS(LCASE(STR(?label)), "axum") ||
    CONTAINS(LCASE(STR(?label)), "ethiopic") ||
    CONTAINS(LCASE(STR(?label)), "ge'ez") ||
    CONTAINS(LCASE(STR(?label)), "geez")
  )
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{safe.lower()}")) ||
         CONTAINS(LCASE("{safe.lower()}"), "ethiopia"))
  OPTIONAL {{
    ?object crm:P138i_has_representation ?rep .
    ?rep crm:P138i_is_representation_of ?image .
  }}
}} LIMIT {limit}
"""
    try:
        r = session.get(
            BM_SPARQL_API,
            params={"query": sparql},
            timeout=BM_REQUEST_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        rows = data.get("results", {}).get("bindings", [])
        out: list[dict[str, str]] = []
        for row in rows:
            obj_uri = row.get("object", {}).get("value", "")
            label = row.get("label", {}).get("value", "")
            image = row.get("image", {}).get("value", "")
            if not obj_uri:
                continue
            oid = obj_uri.rstrip("/").split("/")[-1]
            out.append({
                "id": oid,
                "label": label,
                "image_url": image,
                "object_url": obj_uri,
            })
        return out
    except Exception as exc:
        logger.debug(f"BM SPARQL unavailable: {exc}")
        return []


def _bm_rest_search(
    session: requests.Session,
    query_text: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    """
    Search British Museum REST id/object endpoint (user-specified API).

    Parses JSON, JSON-LD @graph, or a list of object URIs.
    """
    try:
        r = session.get(
            BM_SEARCH_API,
            params={"query": query_text, "limit": limit},
            timeout=BM_REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        if r.status_code != 200:
            return []
        ctype = (r.headers.get("content-type") or "").lower()
        if "json" not in ctype:
            return []
        data = r.json()
        out: list[dict[str, str]] = []

        def _add_item(item: dict) -> None:
            uri = (
                item.get("@id")
                or item.get("id")
                or item.get("uri")
                or item.get("object")
                or ""
            )
            label = (
                item.get("label")
                or item.get("title")
                or item.get("prefLabel")
                or item.get("name")
                or ""
            )
            image = (
                item.get("image")
                or item.get("thumbnail")
                or item.get("primaryImage")
                or ""
            )
            if isinstance(image, dict):
                image = image.get("url") or image.get("@id") or ""
            if uri:
                oid = str(uri).rstrip("/").split("/")[-1]
                out.append({
                    "id": oid,
                    "label": str(label),
                    "image_url": str(image),
                    "object_url": str(uri),
                })

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    _add_item(item)
                elif isinstance(item, str):
                    oid = item.rstrip("/").split("/")[-1]
                    out.append({
                        "id": oid,
                        "label": query_text,
                        "image_url": "",
                        "object_url": item,
                    })
        elif isinstance(data, dict):
            graph = data.get("@graph") or data.get("results") or data.get("items")
            if isinstance(graph, list):
                for item in graph:
                    if isinstance(item, dict):
                        _add_item(item)
            else:
                _add_item(data)
        return out
    except Exception as exc:
        logger.debug(f"BM REST search unavailable: {exc}")
        return []


def _bm_fetch_object_image(
    session: requests.Session,
    object_url: str,
) -> tuple[str, str]:
    """
    Fetch object JSON from BM and extract title + best image URL.

    Returns (title_text, image_url).
    """
    json_url = object_url.rstrip("/") + ".json"
    try:
        r = session.get(json_url, timeout=BM_REQUEST_TIMEOUT)
        if r.status_code != 200:
            return "", ""
        data = r.json()
        texts: list[str] = []
        image_url = ""

        def _walk(node: object) -> None:
            nonlocal image_url
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = k.lower()
                    if kl in ("label", "title", "preflabel", "name") and isinstance(v, str):
                        texts.append(v)
                    if kl in ("image", "thumbnail", "contenturl", "url") and isinstance(v, str):
                        if any(x in v.lower() for x in (".jpg", ".jpeg", ".png", "/media/")):
                            if not image_url:
                                image_url = v
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(data)
        return " ".join(texts), image_url
    except Exception:
        return "", ""


def fetch_british_museum(
    session: requests.Session,
    class_name: str,
    class_dir: Path,
    query: str,
    need: int,
    metadata: dict[str, ImageMeta],
    tracker: ClassTracker,
    seen_ids: set[str],
) -> int:
    """
    Search British Museum collection API; download bm_{class}_{NNN}.jpg.

    Tries REST search first, then SPARQL. Skips gracefully if BM is unreachable.
    """
    downloaded = 0
    source_abbr = SOURCE_ABBR["britishmuseum"]

    candidates = _bm_rest_search(session, query, limit=100)
    if not candidates:
        candidates = _bm_sparql_search(session, query, limit=100)
    if not candidates and query != MUSEUM_BASE_QUERIES["britishmuseum"]:
        candidates = _bm_rest_search(
            session, MUSEUM_BASE_QUERIES["britishmuseum"], limit=100,
        )
    if not candidates:
        logger.warning(f"British Museum search returned nothing ({query})")
        return 0

    for item in candidates:
        if downloaded >= need:
            break
        sid = item.get("id", "")
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)

        label = item.get("label", "")
        image_url = item.get("image_url", "")
        object_url = item.get("object_url", "")

        if not image_url and object_url:
            extra_label, extra_img = _bm_fetch_object_image(session, object_url)
            label = label or extra_label
            image_url = extra_img

        if not image_url:
            continue

        img_r = _get_with_retry(session, image_url, label=f"bm-img:{sid}")
        if img_r is None:
            continue
        raw = img_r.content

        vr = validate_image_bytes(raw)
        if not vr.ok:
            logger.debug(f"BM reject {sid}: {vr.reason}")
            continue

        bm_text = f"{label} {query} British Museum Ethiopia"
        save_class, gate_reason = _label_gate(bm_text, class_name)
        if save_class is None:
            logger.debug(f"BM label reject {sid}: {gate_reason}")
            continue

        ok, why = tracker.can_add(save_class, sid, vr.ahash)
        if not ok:
            logger.debug(f"BM dup skip {sid}: {why}")
            continue

        save_dir = ARTEFACT_CLASSES_DIR / save_class
        save_dir.mkdir(parents=True, exist_ok=True)
        idx = _next_file_index(save_dir, source_abbr, save_class)
        fname = f"{source_abbr}_{save_class}_{idx:03d}.jpg"
        if _save_jpeg(save_dir, raw, fname) is None:
            continue

        tracker.register(save_class, sid, vr.ahash)
        page_url = object_url or f"https://www.britishmuseum.org/collection/object/{sid}"
        note = f"bm query={query!r}; id={sid}; title={label[:60]}"
        if gate_reason and gate_reason != "ok":
            note += f"; gate={gate_reason}"
        metadata[fname] = ImageMeta(
            filename=fname,
            class_name=save_class,
            source_url=page_url,
            license="British Museum (research/non-commercial)",
            width_px=vr.width,
            height_px=vr.height,
            notes=note,
        )
        if save_class == class_name:
            downloaded += 1
            logger.info(
                f"  [{class_name}] bm +1 ({downloaded}/{need}) {fname}",
            )
        else:
            logger.info(
                f"  [{class_name}] bm reroute → {save_class}: {fname}",
            )

    return downloaded


def fetch_met(
    session: requests.Session,
    class_name: str,
    class_dir: Path,
    query: str,
    need: int,
    metadata: dict[str, ImageMeta],
    tracker: ClassTracker,
    seen_ids: set[str],
) -> int:
    """
    Search Met Open Access; download primary images as met_{class}_{NNN}.jpg.
    """
    downloaded = 0
    source_abbr = SOURCE_ABBR["met"]

    r = _get_with_retry(
        session,
        MET_SEARCH_API,
        params={"q": query, "hasImages": "true", "isPublicDomain": "true"},
        label=f"met-search:{query}",
    )
    if r is None:
        logger.warning(f"Met search failed ({query})")
        return 0
    object_ids = r.json().get("objectIDs") or []

    if not object_ids:
        return 0

    for oid in object_ids:
        if downloaded >= need:
            break
        sid = str(oid)
        if sid in seen_ids:
            continue
        seen_ids.add(sid)

        time.sleep(REQUEST_DELAY_S)
        obj_r = _get_with_retry(
            session,
            f"{MET_OBJECT_API}{oid}",
            label=f"met-obj:{oid}",
        )
        if obj_r is None:
            continue
        obj = obj_r.json()

        if not obj.get("isPublicDomain"):
            continue

        url = obj.get("primaryImage") or obj.get("primaryImageSmall")
        if not url:
            continue

        img_r = _get_with_retry(session, url, label=f"met-img:{oid}")
        if img_r is None:
            continue
        raw = img_r.content

        vr = validate_image_bytes(raw)
        if not vr.ok:
            logger.debug(f"Met reject {oid}: {vr.reason}")
            continue

        met_text = build_met_context(obj)
        save_class, gate_reason = _label_gate(met_text, class_name)
        if save_class is None:
            logger.debug(f"Met label reject {oid}: {gate_reason}")
            continue

        ok, why = tracker.can_add(save_class, sid, vr.ahash)
        if not ok:
            logger.debug(f"Met dup skip {oid}: {why}")
            continue

        save_dir = ARTEFACT_CLASSES_DIR / save_class
        save_dir.mkdir(parents=True, exist_ok=True)
        idx = _next_file_index(save_dir, source_abbr, save_class)
        fname = f"{source_abbr}_{save_class}_{idx:03d}.jpg"
        if _save_jpeg(save_dir, raw, fname) is None:
            continue

        tracker.register(save_class, sid, vr.ahash)
        obj_url = obj.get(
            "objectURL",
            f"https://www.metmuseum.org/art/collection/search/{oid}",
        )
        note = f"met query={query!r}; objectID={oid}; title={obj.get('title', '')[:60]}"
        if gate_reason and gate_reason != "ok":
            note += f"; gate={gate_reason}"
        metadata[fname] = ImageMeta(
            filename=fname,
            class_name=save_class,
            source_url=obj_url,
            license="Met Open Access (CC0)",
            width_px=vr.width,
            height_px=vr.height,
            notes=note,
        )
        if save_class == class_name:
            downloaded += 1
            logger.info(
                f"  [{class_name}] met +1 ({downloaded}/{need}) {fname}",
            )
        else:
            logger.info(
                f"  [{class_name}] met reroute → {save_class}: {fname}",
            )

    return downloaded


def fetch_smithsonian(
    session: requests.Session,
    class_name: str,
    class_dir: Path,
    query: str,
    need: int,
    metadata: dict[str, ImageMeta],
    tracker: ClassTracker,
    seen_ids: set[str],
) -> int:
    """
    Search Smithsonian Open Access and download CC0 object images.

    Uses si_{class}_{NNN}.jpg naming; Open Access media URLs only.
    """
    downloaded = 0
    source_abbr = SOURCE_ABBR["smithsonian"]

    r = _get_with_retry(
        session,
        SI_SEARCH_API,
        params={
            "q": query,
            "rows": min(50, need + 10),
            "api_key": SMITHSONIAN_API_KEY,
        },
        label=f"si-search:{query}",
    )
    if r is None:
        logger.warning(f"Smithsonian search failed ({query})")
        return 0

    rows = r.json().get("response", {}).get("rows", [])
    if not rows:
        return 0

    for row in rows:
        if downloaded >= need:
            break
        sid = row.get("id", "")
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)

        time.sleep(REQUEST_DELAY_S)
        obj_r = _get_with_retry(
            session,
            SI_CONTENT_API.format(id=sid),
            label=f"si-content:{sid}",
        )
        if obj_r is None:
            continue
        content = obj_r.json().get("response", {}).get("content", {})
        online = (
            content.get("descriptiveNonRepeating", {})
            .get("online_media", {})
        )
        media_list = online.get("media") or []
        url = None
        for m in media_list:
            if m.get("type") == "Images":
                url = m.get("content")
                break
        if not url:
            continue

        img_r = _get_with_retry(session, url, label=f"si-img:{sid}")
        if img_r is None:
            continue
        raw = img_r.content

        vr = validate_image_bytes(raw)
        if not vr.ok:
            logger.debug(f"SI reject {sid}: {vr.reason}")
            continue

        title = (
            content.get("descriptiveNonRepeating", {})
            .get("title", {})
            .get("content", f"SI {sid}")
        )
        save_class, gate_reason = _label_gate(f"{title} {query}", class_name)
        if save_class is None:
            logger.debug(f"SI label reject {sid}: {gate_reason}")
            continue

        ok, why = tracker.can_add(save_class, sid, vr.ahash)
        if not ok:
            continue

        save_dir = ARTEFACT_CLASSES_DIR / save_class
        save_dir.mkdir(parents=True, exist_ok=True)
        idx = _next_file_index(save_dir, source_abbr, save_class)
        fname = f"{source_abbr}_{save_class}_{idx:03d}.jpg"
        if _save_jpeg(save_dir, raw, fname) is None:
            continue

        tracker.register(save_class, sid, vr.ahash)
        note = f"si query={query!r}; id={sid}; title={title[:80]}"
        if gate_reason and gate_reason != "ok":
            note += f"; gate={gate_reason}"
        metadata[fname] = ImageMeta(
            filename=fname,
            class_name=save_class,
            source_url=f"https://www.si.edu/object/{sid}",
            license="Smithsonian Open Access (CC0)",
            width_px=vr.width,
            height_px=vr.height,
            notes=note,
        )
        if save_class == class_name:
            downloaded += 1
            logger.info(
                f"  [{class_name}] si +1 ({downloaded}/{need}) {fname}",
            )
        else:
            logger.info(
                f"  [{class_name}] si reroute → {save_class}: {fname}",
            )

    return downloaded


def _ordered_queries(
    class_name: str,
    *,
    museums_only: bool = False,
) -> list[tuple[str, str]]:
    """
    Return search queries for one class.

    Default order: British Museum → Smithsonian → Met → Wikimedia.
    With museums_only, skip Wikimedia (user-specified museum APIs only).
    """
    queries = CLASS_SEARCH_QUERIES.get(class_name, [])
    priority = {"britishmuseum": 0, "smithsonian": 1, "met": 2, "wikimedia": 3}
    ordered = sorted(queries, key=lambda x: priority.get(x[0], 9))
    if museums_only:
        ordered = [q for q in ordered if q[0] in ("britishmuseum", "smithsonian", "met")]
    return ordered


def download_class(
    session: requests.Session,
    class_name: str,
    target: int,
    metadata: dict[str, ImageMeta],
    tracker: ClassTracker,
    dry_run: bool = False,
    museums_only: bool = False,
) -> int:
    """Fill one class folder to `target` validated images."""
    class_dir = ARTEFACT_CLASSES_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    current = _count_images(class_dir)
    if current >= target:
        logger.info(f"{class_name}: {current} >= {target}, skip download")
        return current

    if dry_run:
        logger.info(
            f"{class_name}: {current} images, need {target - current} more (dry-run)",
        )
        return current

    logger.info(
        f"{class_name}: {current} images, downloading {target - current} more "
        f"(target {target})...",
    )

    seen_wiki: set[str] = set()
    seen_met: set[str] = set()
    seen_si: set[str] = set()
    seen_bm: set[str] = set()

    for source, query in _ordered_queries(class_name, museums_only=museums_only):
        if _count_images(class_dir) >= target:
            break
        still_need = target - _count_images(class_dir)
        if still_need <= 0:
            break

        if source == "britishmuseum":
            fetch_british_museum(
                session, class_name, class_dir, query, still_need,
                metadata, tracker, seen_bm,
            )
        elif source == "wikimedia":
            fetch_wikimedia(
                session, class_name, class_dir, query, still_need,
                metadata, tracker, seen_wiki,
            )
        elif source == "met":
            fetch_met(
                session, class_name, class_dir, query, still_need,
                metadata, tracker, seen_met,
            )
        elif source == "smithsonian":
            fetch_smithsonian(
                session, class_name, class_dir, query, still_need,
                metadata, tracker, seen_si,
            )
        if REQUEST_DELAY_S > 0:
            time.sleep(REQUEST_DELAY_S)

    final = _count_images(class_dir)
    min_req = ARTEFACT_CLASS_MIN_COUNTS.get(class_name, 0)
    if final < min_req:
        logger.warning(
            f"{class_name}: {final}/{min_req} minimum — "
            "try more queries or manual turntable photos",
        )
    return final


def _source_breakdown(metadata: dict[str, ImageMeta]) -> dict[str, int]:
    """Count images by source prefix (wiki, met, legacy wiki_File_, etc.)."""
    counts: dict[str, int] = defaultdict(int)
    for fn in metadata:
        if fn.startswith("wiki_") and "_File_" not in fn:
            counts["wiki"] += 1
        elif fn.startswith("met_"):
            counts["met"] += 1
        elif fn.startswith("si_"):
            counts["si"] += 1
        elif fn.startswith("bm_"):
            counts["bm"] += 1
        elif fn.startswith("wiki_File") or fn.startswith("wiki_"):
            counts["wiki_legacy"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def write_sources_md(
    counts: dict[str, int],
    metadata: dict[str, ImageMeta],
    targets: dict[str, int],
) -> None:
    """Document APIs, counts, and sample provenance in SOURCES.md."""
    lines = [
        "# Artefact classification training images — sources",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Educational dataset for **WRO 2026 AXUM rover** (turntable top-down).",
        "**Scope:** Ethiopian & Horn of Africa cultural artefacts (not Aksum-only).",
        "Licences: CC BY, CC BY-SA, CC0, public domain only (no NC).",
        "",
        "Label rules: `src/object_detection/artefact_label_rules.py`",
        "Cleanup: `python scripts/cleanup_artefact_dataset.py`",
        "",
        "## Per-class counts (Section 6)",
        "",
        "| Class | Count | Minimum | Ideal | Status |",
        "|-------|-------|---------|-------|--------|",
    ]
    for cls in OBJ_CLASSES:
        n = counts.get(cls, 0)
        mn = ARTEFACT_CLASS_MIN_COUNTS.get(cls, 0)
        ideal = ARTEFACT_CLASS_IDEAL_COUNTS.get(cls, 0)
        tgt = targets.get(cls, mn)
        if n >= ideal:
            st = "ideal"
        elif n >= mn:
            st = "minimum OK"
        else:
            st = "SHORT"
        lines.append(f"| `{cls}` | {n} | {mn} | {ideal} | {st} ({tgt} run) |")

    total = sum(counts.get(c, 0) for c in OBJ_CLASSES)
    lines.extend([
        "",
        f"**Total images:** {total} "
        f"(minimum {ARTEFACT_DATASET_MIN_TOTAL}, ideal {ARTEFACT_DATASET_IDEAL_TOTAL})",
        "",
        "## APIs used",
        "",
        "1. [British Museum Collection API](https://collection.britishmuseum.org/id/object) "
        "— Ethiopia search + SPARQL fallback",
        "2. [Smithsonian Open Access API](https://api.si.edu/openaccess/) — CC0 media",
        "3. [Met Collection API](https://collectionapi.metmuseum.org/) "
        "— `isPublicDomain=true` objects only",
        "4. [Wikimedia Commons API](https://commons.wikimedia.org/w/api.php) "
        "— File namespace, licence filter (fallback when museums short)",
        "",
        "**Not used:** Pinterest, Getty, Shutterstock, unfiltered image search.",
        "",
        "## Quality filters",
        "",
        f"- Minimum {ARTEFACT_MIN_IMAGE_PX}×{ARTEFACT_MIN_IMAGE_PX} px, JPEG/PNG",
        f"- Coverage heuristic ≥ {ARTEFACT_MIN_COVERAGE_RATIO:.0%} non-background pixels",
        f"- Laplacian sharpness ≥ {ARTEFACT_MIN_SHARPNESS_VAR}",
        f"- Max {ARTEFACT_MAX_NEAR_DUPES_PER_SOURCE} images per source artefact",
        "",
        "## Download script",
        "",
        "```bash",
        "cd C:\\Users\\Len\\Documents\\Programming\\Projects\\AXUM",
        ".\\venv\\Scripts\\activate",
        "python scripts/download_artefact_images.py",
        "python scripts/download_artefact_images.py --ideal",
        "python scripts/download_artefact_images.py --class coin",
        "python scripts/download_artefact_images.py --report-only",
        "```",
        "",
        "## metadata.csv",
        "",
        f"Row count: {len(metadata)} (see `{ARTEFACT_METADATA_CSV.name}`)",
        "",
        "## Sample provenance",
        "",
    ])

    by_class: dict[str, list[ImageMeta]] = {c: [] for c in OBJ_CLASSES}
    for m in metadata.values():
        if m.class_name in by_class:
            by_class[m.class_name].append(m)

    for cls in OBJ_CLASSES:
        entries = by_class[cls][:5]
        lines.append(f"### `{cls}`")
        lines.append("")
        if not entries:
            lines.append("_No metadata entries yet._")
        else:
            for e in entries:
                lines.append(
                    f"- **{e.filename}** — {e.license} "
                    f"({e.width_px}×{e.height_px}) [source]({e.source_url})",
                )
        lines.append("")

    breakdown = _source_breakdown(metadata)
    lines.append("## Source prefix breakdown (metadata rows)")
    lines.append("")
    for k, v in sorted(breakdown.items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    out = ARTEFACT_CLASSES_DIR / "SOURCES.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out}")


def print_balance_report(
    counts: dict[str, int],
    metadata: dict[str, ImageMeta],
    targets: dict[str, int],
) -> bool:
    """
    Section 6 balance report: per-class counts, total, sources, ratio check.

    Returns True if every class meets minimum and ratio ≤ ARTEFACT_MAX_CLASS_RATIO.
    """
    print("\n" + "=" * 70)
    print("ARTEFACT DATASET — SECTION 6 BALANCE REPORT")
    print("=" * 70)

    all_min_ok = True
    total = 0
    class_mins: list[int] = []

    for cls in OBJ_CLASSES:
        n = counts.get(cls, 0)
        total += n
        mn = ARTEFACT_CLASS_MIN_COUNTS.get(cls, 0)
        ideal = ARTEFACT_CLASS_IDEAL_COUNTS.get(cls, 0)
        tgt = targets.get(cls, mn)
        if n >= mn:
            mark = "OK"
        else:
            mark = "SHORT"
            all_min_ok = False
        short = max(0, mn - n)
        print(
            f"  [{mark}] {cls:<22} {n:4d}  "
            f"(min {mn}, ideal {ideal}, target run {tgt}"
            + (f", need +{short}" if short else "") + ")",
        )
        if n > 0:
            class_mins.append(n)

    print(f"\n  Total images:     {total}")
    print(f"  Minimum required: {ARTEFACT_DATASET_MIN_TOTAL}")
    print(f"  Ideal target:     {ARTEFACT_DATASET_IDEAL_TOTAL}")
    print(f"  metadata.csv rows: {len(metadata)}")

    breakdown = _source_breakdown(metadata)
    print("\n  Sources (from metadata filenames):")
    for k, v in sorted(breakdown.items()):
        print(f"    {k:<14} {v}")

    if class_mins:
        ratio = max(class_mins) / max(min(class_mins), 1)
        ratio_ok = ratio <= ARTEFACT_MAX_CLASS_RATIO
        print(
            f"\n  Class balance ratio (max/min): {ratio:.2f} "
            f"(limit {ARTEFACT_MAX_CLASS_RATIO}) "
            f"{'OK' if ratio_ok else 'IMBALANCED'}",
        )
        all_min_ok = all_min_ok and ratio_ok
    else:
        print("\n  Class balance: no images")
        all_min_ok = False

    print("=" * 70)
    return all_min_ok


# Second-stage YOLO: era + denomination (folder = class name)
COIN_SUBCLASS_SEARCH: dict[str, list[tuple[str, str]]] = {
    "coin_aksumite": [
        ("wikimedia", "Aksumite coin obverse"),
        ("wikimedia", "Ezana coin gold"),
        ("wikimedia", "King Kaleb coin"),
    ],
    "coin_menelik": [
        ("wikimedia", "Menelik II coin"),
        ("wikimedia", "Emperor Menelik coin Ethiopia"),
    ],
    "coin_haile_selassie": [
        ("wikimedia", "Haile Selassie coin"),
        ("wikimedia", "Ethiopia Haile Selassie coin"),
    ],
    "coin_modern_birr": [
        ("wikimedia", "Ethiopia 1 birr coin"),
        ("wikimedia", "Ethiopian birr coin"),
        ("wikimedia", "Ethiopia 5 birr coin"),
        ("wikimedia", "Ethiopia 10 birr coin"),
    ],
    "coin_modern_cent": [
        ("wikimedia", "Ethiopia 25 cent coin"),
        ("wikimedia", "Ethiopia 50 cent coin"),
        ("wikimedia", "Ethiopian cent coin"),
    ],
    "coin_modern_other": [
        ("wikimedia", "Ethiopian coin circulation"),
        ("wikimedia", "National Bank of Ethiopia coin"),
    ],
}


def _coin_subclass_gate(text: str) -> tuple[bool, str]:
    """
    Gate for coin-subtype downloads: must be Ethiopian coin, not cross/painting.

    Args:
        text: Title + filename combined

    Returns:
        (allowed, reason)
    """
    from src.object_detection.artefact_label_rules import (
        COIN_REJECT,
        is_globally_rejected,
    )

    t = text.lower()
    rej, reason = is_globally_rejected(text)
    if rej:
        return False, reason
    if _any_kw(t, FOREIGN_COIN_REJECT) and not is_ethiopia_related(text):
        return False, "foreign_coin"
    if _any_kw(t, COIN_REJECT):
        return False, "cross_not_coin"
    if not _any_kw(t, COIN_POSITIVE) and not is_ethiopia_related(text):
        return False, "not_ethiopian_coin"
    return True, "ok"


def _any_kw(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def download_coin_subtypes(
    session: requests.Session,
    targets: dict[str, int] | None = None,
    metadata: dict[str, ImageMeta] | None = None,
) -> dict[str, int]:
    """
    Download images into data/coin_subtypes/{subtype}/ for coin-detail YOLO.

    Args:
        session: HTTP session
        targets: Per-subtype image counts (default COIN_SUBCLASS_TARGETS)
        metadata: Optional shared metadata dict (uses coin_subtypes_metadata.csv)

    Returns:
        Per-subtype counts on disk
    """
    targets = targets or dict(COIN_SUBCLASS_TARGETS)
    meta_path = COIN_SUBTYPES_DIR / "metadata.csv"
    metadata = metadata if metadata is not None else _load_metadata_from(meta_path)
    tracker = ClassTracker()
    counts: dict[str, int] = {}

    COIN_SUBTYPES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Coin subtype download → {COIN_SUBTYPES_DIR}")

    for subtype, queries in COIN_SUBCLASS_SEARCH.items():
        target = targets.get(subtype, COIN_SUBCLASS_MIN_IMAGES)
        class_dir = COIN_SUBTYPES_DIR / subtype
        class_dir.mkdir(parents=True, exist_ok=True)
        current = _count_images(class_dir)
        if current >= target:
            counts[subtype] = current
            continue

        seen: set[str] = set()
        for source, query in queries:
            if _count_images(class_dir) >= target:
                break
            still = target - _count_images(class_dir)

            if source != "wikimedia":
                continue

            downloaded = 0
            offset = 0
            while downloaded < still:
                params = {
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrsearch": f"filetype:bitmap {query} {WIKIMEDIA_NEG_KEYWORDS}",
                    "gsrnamespace": 6,
                    "gsrlimit": min(50, still - downloaded + 10),
                    "gsroffset": offset,
                    "prop": "imageinfo",
                    "iiprop": "url|mime|extmetadata",
                    "iiurlwidth": 1024,
                }
                r = _get_with_retry(
                    session, WIKIMEDIA_API, params=params,
                    label=f"coin-sub:{subtype}",
                )
                if r is None:
                    break
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    break

                for page in pages.values():
                    if downloaded >= still:
                        break
                    title = page.get("title", "")
                    if not title or title in seen:
                        continue
                    seen.add(title)
                    ok, why = _coin_subclass_gate(title)
                    if not ok:
                        logger.debug(f"Coin sub skip {title}: {why}")
                        continue

                    ii = (page.get("imageinfo") or [{}])[0]
                    if ii.get("mime") not in ("image/jpeg", "image/png"):
                        continue
                    lic = (
                        (ii.get("extmetadata") or {})
                        .get("LicenseShortName", {})
                        .get("value", "")
                    )
                    if not _license_ok(lic):
                        continue
                    url = ii.get("thumburl") or ii.get("url")
                    if not url:
                        continue

                    time.sleep(WIKIMEDIA_IMAGE_DELAY_S)
                    img_r = _get_with_retry(session, url, label=f"coin-img:{title[:30]}")
                    if img_r is None:
                        continue
                    vr = validate_image_bytes(img_r.content)
                    if not vr.ok:
                        continue
                    ok_dup, _ = tracker.can_add(subtype, title, vr.ahash)
                    if not ok_dup:
                        continue

                    abbr = "wiki"
                    idx = _next_file_index(class_dir, abbr, subtype)
                    fname = f"{abbr}_{subtype}_{idx:03d}.jpg"
                    if _save_jpeg(class_dir, img_r.content, fname) is None:
                        continue
                    tracker.register(subtype, title, vr.ahash)
                    metadata[fname] = ImageMeta(
                        filename=fname,
                        class_name=subtype,
                        source_url=f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                        license=lic,
                        width_px=vr.width,
                        height_px=vr.height,
                        notes=f"coin_subtype query={query!r}",
                    )
                    downloaded += 1
                    logger.info(f"  [{subtype}] +1 {fname}")

                cont = data.get("continue", {}).get("gsroffset")
                if cont is None:
                    break
                offset = cont
                time.sleep(REQUEST_DELAY_S)

        counts[subtype] = _count_images(class_dir)

    _save_metadata_to(meta_path, metadata)
    return counts


def _load_metadata_from(path: Path) -> dict[str, ImageMeta]:
    """Load metadata CSV from arbitrary path."""
    out: dict[str, ImageMeta] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if fn:
                out[fn] = ImageMeta(
                    filename=fn,
                    class_name=row.get("class", ""),
                    source_url=row.get("source_url", ""),
                    license=row.get("license", ""),
                    width_px=int(row.get("width_px") or 0),
                    height_px=int(row.get("height_px") or 0),
                    notes=row.get("notes", ""),
                )
    return out


def _save_metadata_to(path: Path, rows: dict[str, ImageMeta]) -> None:
    """Write metadata CSV to given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows.values(), key=lambda r: (r.class_name, r.filename))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({
                "filename": r.filename,
                "class": r.class_name,
                "source_url": r.source_url,
                "license": r.license,
                "width_px": r.width_px,
                "height_px": r.height_px,
                "notes": r.notes,
            })


def _resolve_targets(use_ideal: bool, override: int | None = None) -> dict[str, int]:
    """Per-class download targets (minimum, ideal, or CLI override)."""
    if override is not None:
        return {cls: override for cls in OBJ_CLASSES}
    if use_ideal:
        return dict(ARTEFACT_CLASS_IDEAL_COUNTS)
    return dict(ARTEFACT_CLASS_MIN_COUNTS)


def main() -> int:
    """CLI: download open-licence artefact images until targets met."""
    global REQUEST_DELAY_S, WIKIMEDIA_IMAGE_DELAY_S

    parser = argparse.ArgumentParser(
        description="Download artefact classifier images (BM + SI + Met + Wikimedia).",
    )
    parser.add_argument(
        "--ideal",
        action="store_true",
        help="Aim for ideal per-class counts (not just minimum)",
    )
    parser.add_argument(
        "--class",
        dest="only_class",
        choices=OBJ_CLASSES,
        default=None,
        help="Download a single class only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report gaps only; no downloads",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print balance report from disk; no downloads",
    )
    parser.add_argument(
        "--coin-subtypes",
        action="store_true",
        help="Download coin era/denomination images to data/coin_subtypes/",
    )
    parser.add_argument(
        "--coin-subtypes-only",
        action="store_true",
        help="Only run coin subtype download (skip 5-class artefact pass)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        metavar="N",
        help="Override per-class image target (e.g. 80 for all classes)",
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Disable inter-request delays (museum API runs)",
    )
    parser.add_argument(
        "--museums-only",
        action="store_true",
        help="Use only British Museum, Smithsonian, and Met (no Wikimedia)",
    )
    args = parser.parse_args()

    if args.no_delay:
        REQUEST_DELAY_S = 0.0
        WIKIMEDIA_IMAGE_DELAY_S = 0.0

    ARTEFACT_CLASSES_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata()
    targets = _resolve_targets(args.ideal, args.target)
    classes = [args.only_class] if args.only_class else list(OBJ_CLASSES)

    if args.report_only:
        counts = {
            c: _count_images(ARTEFACT_CLASSES_DIR / c) for c in OBJ_CLASSES
        }
        print_balance_report(counts, metadata, targets)
        write_sources_md(counts, metadata, targets)
        return 0

    session = _session()

    if args.coin_subtypes or args.coin_subtypes_only:
        logger.info("Coin subtype download (Menelik, Haile Selassie, birr, cent, …)")
        sub_counts = download_coin_subtypes(session)
        for st, n in sub_counts.items():
            logger.info(f"  {st}: {n}")
        if args.coin_subtypes_only:
            return 0

    logger.info("AXUM artefact image downloader")
    logger.info(f"Output: {ARTEFACT_CLASSES_DIR}")
    logger.info(f"Targets: {targets}")

    tracker = ClassTracker()
    counts: dict[str, int] = {}

    if not args.dry_run:
        for cls in classes:
            counts[cls] = download_class(
                session, cls, targets[cls], metadata, tracker,
                dry_run=False, museums_only=args.museums_only,
            )
            _save_metadata(metadata)
    else:
        for cls in classes:
            counts[cls] = download_class(
                session, cls, targets[cls], metadata, tracker,
                dry_run=True, museums_only=args.museums_only,
            )

    for cls in OBJ_CLASSES:
        if cls not in counts:
            counts[cls] = _count_images(ARTEFACT_CLASSES_DIR / cls)

    backfill_metadata_from_disk(metadata)
    if not args.dry_run:
        _save_metadata(metadata)

    write_sources_md(counts, metadata, targets)
    ready = print_balance_report(counts, metadata, targets)

    if args.dry_run:
        return 0
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
