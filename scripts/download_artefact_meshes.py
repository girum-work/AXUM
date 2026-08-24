"""
AXUM ROVER — CC0 Artefact Mesh Downloader
==========================================
Replaces the procedural placeholder meshes in ``scans/meshes/`` with real
museum 3D scans, so the catalogue grid and mesh-pipeline work are exercised
against actual artefact geometry instead of lathed spheres.

Source: Smithsonian Open Access (``online_media_type:"3D Models"``), CC0 only.

Why Smithsonian and why GLB:
  The Met's Open Access API publishes images, not meshes. Smithsonian is the
  only large CC0 collection that publishes downloadable 3D, and it publishes
  **glTF (.glb) and USDZ — no OBJ and no PLY**. That is a fact about the
  collection, not a preference: a survey of 300 records found 212 ``glb``,
  350 ``zip`` (gltf packages), 114 ``usdz`` and zero ``obj``/``ply``. So the
  viewers had to learn GLB rather than the meshes being converted on the way
  in; converting would throw away the PBR materials that make these scans
  worth having.

Every download is recorded with its Smithsonian record id, title, unit and
licence in ``provenance.json`` beside the mesh, and collectively in
``scans/meshes/SOURCES.md``. These are other institutions' scans of other
institutions' objects — they are reference geometry, not AXUM captures, and
the catalogue grid labels them accordingly.

Run from project root (venv activated):
  python scripts/download_artefact_meshes.py --list
  python scripts/download_artefact_meshes.py --dry-run
  python scripts/download_artefact_meshes.py --per-query 2
  python scripts/download_artefact_meshes.py --query "Ethiopia" --per-query 5

Writes:
  scans/meshes/SI-<slug>/model.glb
  scans/meshes/SI-<slug>/provenance.json
  scans/meshes/SOURCES.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MESH_DIR, SMITHSONIAN_API_KEY

SI_SEARCH_API = "https://api.si.edu/openaccess/api/v1.0/search"

# Smithsonian's free-text index is the only way in -- there is no
# "is an archaeological artefact" facet. These queries are AND-ed with the
# 3D Models media filter and hand-picked to bias toward the object classes
# the rover actually classifies (pottery, coins, carved stone, inscriptions).
DEFAULT_QUERIES: tuple[tuple[str, str], ...] = (
    ("pottery", "pottery vessel"),
    ("pottery", "ceramic jar"),
    ("stone_carving", "stone sculpture"),
    ("stone_carving", "carved stone"),
    ("inscription_fragment", "stele inscription"),
    ("coin", "coin"),
    ("other", "Ethiopia"),
    ("other", "archaeology artifact"),
)

# Low-resolution variants are 1-10 MB against 50-300 MB for full resolution.
# The grid renders every visible card every frame, so the small variant is
# the right default -- pass --full-resolution when the mesh is the subject
# rather than a thumbnail.
PREFERRED_CATEGORIES = ("low resolution", "full resolution", "watertight")
REQUEST_TIMEOUT = 60
POLITE_DELAY_SEC = 0.4
SEARCH_MAX_ATTEMPTS = 4


@dataclass
class MeshCandidate:
    """One downloadable CC0 mesh resolved from a Smithsonian record."""

    object_id: str
    record_id: str
    title: str
    unit: str
    category: str
    file_type: str
    file_size: int
    url: str
    record_url: str
    artefact_class: str


def _slugify(text: str, limit: int = 38) -> str:
    """
    Build a filesystem- and URL-safe folder suffix from a record title.

    Args:
        text: Raw museum title
        limit: Maximum slug length

    Returns:
        Uppercase hyphenated slug
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").upper()
    return (cleaned[:limit].rstrip("-")) or "UNTITLED"


def _resource_attributes(resource: dict) -> dict:
    """
    Flatten a Smithsonian resource's ``attributes`` list into one dict.

    The API returns attributes as a list of single-key dicts
    (``[{"FILE_TYPE": ...}, {"FILE_SIZE": ...}]``) rather than one object,
    so every consumer has to merge them.

    Args:
        resource: One entry from a media record's ``resources``

    Returns:
        Merged attribute dict
    """
    merged: dict = {}
    for attribute in resource.get("attributes") or []:
        if isinstance(attribute, dict):
            merged.update(attribute)
    return merged


def _pick_resource(media: dict, prefer_full: bool) -> dict | None:
    """
    Choose the best downloadable GLB from one media record.

    Args:
        media: A ``online_media.media`` entry
        prefer_full: Prefer full-resolution over low-resolution

    Returns:
        The chosen resource dict, or None when the record has no plain GLB
        (USDZ and zipped glTF packages are skipped -- three.js loads
        neither without extra unpacking)
    """
    order = list(PREFERRED_CATEGORIES)
    if prefer_full:
        order.remove("full resolution")
        order.insert(0, "full resolution")

    ranked: list[tuple[int, int, dict]] = []
    for resource in media.get("resources") or []:
        attributes = _resource_attributes(resource)
        file_type = str(attributes.get("FILE_TYPE", "")).lower()
        if file_type != "glb":
            continue
        category = str(resource.get("category", "")).lower()
        rank = order.index(category) if category in order else len(order)
        ranked.append((rank, int(attributes.get("FILE_SIZE") or 0), resource))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2]


def _search_with_backoff(params: dict, label: str) -> list[dict]:
    """
    Run one search, retrying on HTTP 429.

    api.data.gov throttles DEMO_KEY to a few dozen requests per hour, and a
    full default run is eight queries -- so 429 is the normal failure here,
    not an exceptional one. Backing off turns "no candidates found" (which
    reads like the collection has nothing) into a slower but correct run.

    Args:
        params: Query parameters including the API key
        label: Human-readable query name for log lines

    Returns:
        Result rows, or empty on persistent failure
    """
    delay = 4.0
    for attempt in range(1, SEARCH_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(SI_SEARCH_API, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                if attempt == SEARCH_MAX_ATTEMPTS:
                    logger.warning(f"Rate-limited on {label!r} after {attempt} attempts")
                    return []
                logger.info(f"Rate-limited on {label!r} — retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json().get("response", {}).get("rows", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"Smithsonian search failed ({label}): {exc}")
            return []
    return []


def search_candidates(
    query: str,
    artefact_class: str,
    limit: int,
    prefer_full: bool,
) -> list[MeshCandidate]:
    """
    Search Smithsonian Open Access for CC0 3D records matching a query.

    Args:
        query: Free-text search terms
        artefact_class: AXUM class label to tag results with
        limit: Maximum candidates to return
        prefer_full: Prefer full-resolution meshes

    Returns:
        Resolved candidates, at most ``limit``. Returns empty on any API
        failure -- one dead query should not abort a whole run.
    """
    params = {
        "api_key": SMITHSONIAN_API_KEY,
        "q": f'online_media_type:"3D Models" AND ({query})',
        "rows": 100,
    }
    rows = _search_with_backoff(params, query)
    if not rows:
        return []

    candidates: list[MeshCandidate] = []
    for row in rows:
        if len(candidates) >= limit:
            break
        content = row.get("content", {})
        descriptive = content.get("descriptiveNonRepeating", {})
        for media in descriptive.get("online_media", {}).get("media", []):
            # CC0 only. Records with any other access string are someone
            # else's rights decision and are not ours to redistribute.
            if (media.get("usage") or {}).get("access") != "CC0":
                continue
            resource = _pick_resource(media, prefer_full)
            if not resource:
                continue

            attributes = _resource_attributes(resource)
            title = row.get("title") or "Untitled"
            candidates.append(MeshCandidate(
                object_id=f"SI-{_slugify(title)}",
                record_id=row.get("id", ""),
                title=title,
                unit=descriptive.get("data_source") or row.get("unitCode", ""),
                category=resource.get("category", ""),
                file_type=str(attributes.get("FILE_TYPE", "")),
                file_size=int(attributes.get("FILE_SIZE") or 0),
                url=resource.get("url", ""),
                record_url=media.get("content", ""),
                artefact_class=artefact_class,
            ))
            break

    return candidates


def download_candidate(candidate: MeshCandidate, overwrite: bool) -> bool:
    """
    Download one mesh and write its provenance record.

    Args:
        candidate: Resolved mesh candidate
        overwrite: Re-download even when the folder already has a model

    Returns:
        True when a file was written
    """
    folder = MESH_DIR / candidate.object_id
    target = folder / "model.glb"
    if target.exists() and not overwrite:
        logger.info(f"Skip (exists): {candidate.object_id}")
        return False

    folder.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".glb.part")
    try:
        with requests.get(candidate.url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
    except requests.RequestException as exc:
        logger.warning(f"Download failed for {candidate.object_id}: {exc}")
        partial.unlink(missing_ok=True)
        return False

    # Validate before the file takes its final name: a half-written GLB that
    # is already called model.glb will be picked up by the mesh registry and
    # render as a silently empty card.
    from src.photogrammetry.meshroom import validate_mesh

    ok, message = validate_mesh(partial)
    if not ok:
        logger.warning(f"Rejected {candidate.object_id}: {message}")
        partial.unlink(missing_ok=True)
        return False

    partial.replace(target)

    provenance = asdict(candidate)
    provenance["licence"] = "CC0 (Smithsonian Open Access)"
    provenance["downloaded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance["local_file"] = target.relative_to(MESH_DIR.parent.parent).as_posix()
    provenance["validation"] = message
    with open(folder / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False)

    logger.success(
        f"{candidate.object_id} — {target.stat().st_size / 1e6:.1f} MB "
        f"({candidate.category or 'uncategorised'})"
    )
    return True


def write_sources_manifest() -> Path:
    """
    Regenerate ``scans/meshes/SOURCES.md`` from every provenance.json.

    Returns:
        Path to the written manifest
    """
    rows = []
    for provenance_path in sorted(MESH_DIR.glob("*/provenance.json")):
        with open(provenance_path, encoding="utf-8") as handle:
            rows.append(json.load(handle))

    lines = [
        "# Downloaded artefact meshes",
        "",
        "Third-party 3D scans used as reference geometry in the AXUM catalogue.",
        "These are **not** AXUM rover captures. Regenerate with",
        "`python scripts/download_artefact_meshes.py`.",
        "",
        "| Folder | Title | Holding unit | Licence | Variant | Record |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('object_id', '')}` "
            f"| {row.get('title', '')} "
            f"| {row.get('unit', '')} "
            f"| {row.get('licence', '')} "
            f"| {row.get('category', '')} "
            f"| {row.get('record_url', '')} |"
        )
    lines.append("")
    lines.append(f"{len(rows)} mesh(es). Generated "
                 f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.")

    manifest = MESH_DIR / "SOURCES.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download CC0 artefact meshes from Smithsonian Open Access",
    )
    parser.add_argument(
        "--query", action="append", default=None,
        help="Extra free-text query (repeatable); replaces the default set",
    )
    parser.add_argument(
        "--per-query", type=int, default=2,
        help="Maximum meshes to take per query (default 2)",
    )
    parser.add_argument(
        "--full-resolution", action="store_true",
        help="Prefer full-resolution meshes (tens to hundreds of MB each)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download meshes that already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print candidates without downloading",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Only regenerate SOURCES.md from existing downloads",
    )
    args = parser.parse_args()

    if args.list:
        manifest = write_sources_manifest()
        logger.info(f"Wrote {manifest}")
        return 0

    if SMITHSONIAN_API_KEY == "DEMO_KEY":
        logger.warning(
            "Using DEMO_KEY — Smithsonian rate-limits this hard. Set "
            "SMITHSONIAN_API_KEY in config.py from https://api.data.gov/signup/"
        )

    queries = ([("other", q) for q in args.query] if args.query
               else list(DEFAULT_QUERIES))

    seen: set[str] = set()
    candidates: list[MeshCandidate] = []
    for artefact_class, query in queries:
        found = search_candidates(query, artefact_class, args.per_query, args.full_resolution)
        for candidate in found:
            if candidate.object_id in seen:
                continue
            seen.add(candidate.object_id)
            candidates.append(candidate)
        logger.info(f"{query!r}: {len(found)} candidate(s)")
        time.sleep(POLITE_DELAY_SEC)

    if not candidates:
        logger.error("No CC0 GLB meshes resolved — nothing to download")
        return 1

    total_mb = sum(c.file_size for c in candidates) / 1e6
    logger.info(f"{len(candidates)} mesh(es), ~{total_mb:.0f} MB total")

    if args.dry_run:
        for candidate in candidates:
            logger.info(
                f"  {candidate.object_id:<44} {candidate.file_size / 1e6:6.1f} MB  "
                f"{candidate.category or '-':<16} {candidate.title[:50]}"
            )
        return 0

    written = 0
    for candidate in candidates:
        if download_candidate(candidate, args.overwrite):
            written += 1
        time.sleep(POLITE_DELAY_SEC)

    manifest = write_sources_manifest()
    logger.success(f"Downloaded {written} mesh(es) — provenance in {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
