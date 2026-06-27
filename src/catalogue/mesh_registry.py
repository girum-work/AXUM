# src/catalogue/mesh_registry.py
"""
AXUM ROVER — Mesh Registry
===========================
Links published Meshroom exports to catalogue JSON entries and builds
dashboard-ready metadata (URLs, readiness flags, validation messages).

Author: Axum Rover Team
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CATALOGUE_DIR, MESH_DIR, ROOT_DIR
from src.photogrammetry.meshroom import (
    MeshPublishResult,
    mesh_public_url,
    resolve_mesh_path,
    validate_mesh,
)


def get_mesh_status(object_id: str, mesh_path: str = "") -> dict[str, Any]:
    """
    Inspect whether a catalogue object's mesh is ready for the dashboard.

    Args:
        object_id: Catalogue object ID
        mesh_path: Optional mesh path override; loads from JSON when empty

    Returns:
        Dict with ``mesh_ready``, ``mesh_url``, ``mesh_path``, validation info
    """
    if not mesh_path:
        json_path = CATALOGUE_DIR / f"{object_id}.json"
        if json_path.exists():
            import json
            with open(json_path, encoding="utf-8") as handle:
                mesh_path = json.load(handle).get("mesh_path", "")

    if not mesh_path:
        return {
            "object_id": object_id,
            "mesh_ready": False,
            "mesh_url": None,
            "mesh_path": "",
            "message": "No mesh_path in catalogue entry",
        }

    obj_path = resolve_mesh_path(mesh_path)
    ok, msg = validate_mesh(obj_path) if obj_path.exists() else (False, "File missing")

    return {
        "object_id": object_id,
        "mesh_ready": ok,
        "mesh_url": mesh_public_url(object_id, mesh_path) if ok else None,
        "mesh_path": mesh_path,
        "mesh_filename": obj_path.name,
        "message": msg,
    }


def enrich_catalogue_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Add dashboard mesh fields to one catalogue dict in-place copy.

    Args:
        entry: Raw catalogue JSON dict

    Returns:
        New dict with ``mesh_ready``, ``mesh_url``, ``mesh_filename`` added
    """
    enriched = dict(entry)
    status = get_mesh_status(entry.get("object_id", ""), entry.get("mesh_path", ""))
    enriched["mesh_ready"] = status["mesh_ready"]
    enriched["mesh_url"] = status["mesh_url"]
    enriched["mesh_filename"] = status.get("mesh_filename", "")
    enriched["mesh_status"] = status["message"]
    return enriched


def apply_publish_result(
    record_dict: dict[str, Any],
    result: MeshPublishResult,
) -> dict[str, Any]:
    """
    Merge a ``MeshPublishResult`` into a catalogue record dict.

    Args:
        record_dict: Existing or partial catalogue record
        result:      Published mesh metadata

    Returns:
        Updated record dict with ``mesh_path`` and ``mesh_duration`` set
    """
    updated = dict(record_dict)
    updated["mesh_path"] = result.mesh_path
    updated["mesh_duration"] = round(result.duration_sec, 1)
    if result.warnings:
        errors = list(updated.get("errors") or [])
        errors.extend(result.warnings)
        updated["errors"] = errors
    return updated


def scan_published_meshes() -> list[dict[str, Any]]:
    """
    List all published mesh folders under ``scans/meshes/``.

    Returns:
        List of dicts with object_id, mesh_path, mesh_ready for each folder
    """
    results: list[dict[str, Any]] = []
    if not MESH_DIR.exists():
        return results

    for folder in sorted(MESH_DIR.iterdir()):
        if not folder.is_dir():
            continue
        obj_files = list(folder.glob("*.obj"))
        if not obj_files:
            continue
        rel = obj_files[0].relative_to(ROOT_DIR).as_posix()
        results.append(get_mesh_status(folder.name, rel))
    return results


def sync_catalogue_mesh_paths() -> int:
    """
    Back-fill ``mesh_path`` in catalogue JSON when a published mesh exists
    but the catalogue entry still has an empty path.

    Returns:
        Number of catalogue entries updated
    """
    import json

    updated = 0
    for json_path in sorted(CATALOGUE_DIR.glob("AXUM-OBJ-*.json")):
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)

        if data.get("mesh_path"):
            continue

        object_id = data["object_id"]
        mesh_dir = MESH_DIR / object_id
        obj_files = list(mesh_dir.glob("*.obj")) if mesh_dir.exists() else []
        if not obj_files:
            continue

        rel = obj_files[0].relative_to(ROOT_DIR).as_posix()
        data["mesh_path"] = rel
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        logger.info(f"Synced mesh_path for {object_id} → {rel}")
        updated += 1
    return updated
