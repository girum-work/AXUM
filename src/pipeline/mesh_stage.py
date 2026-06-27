# src/pipeline/mesh_stage.py
"""
AXUM ROVER — Mesh Pipeline Stage
=================================
Callable photogrammetry stage for the mission pipeline and CLI scripts.

Wraps ``src.photogrammetry.meshroom`` and ``CatalogueService`` so a scanned
object automatically gets a published mesh and updated catalogue JSON.

Author: Axum Rover Team
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.catalogue.mesh_registry import apply_publish_result
from src.catalogue.records import ObjectRecord
from src.catalogue.service import CatalogueService
from src.photogrammetry.meshroom import MeshPublishResult, run_photogrammetry


def process_object_mesh(
    object_id: str,
    *,
    skip_meshroom: bool = False,
    export_dir: Path | None = None,
    register: bool = True,
) -> tuple[MeshPublishResult, ObjectRecord | None]:
    """
    Run photogrammetry for one object and optionally update the catalogue.

    Args:
        object_id:     Catalogue ID e.g. AXUM-OBJ-001
        skip_meshroom: Publish from existing export only (no Meshroom subprocess)
        export_dir:    Explicit Meshroom export directory
        register:      When True, write updated record via CatalogueService

    Returns:
        Tuple of (MeshPublishResult, updated ObjectRecord or None)
    """
    result = run_photogrammetry(
        object_id,
        skip_meshroom=skip_meshroom,
        export_dir=export_dir,
    )

    record: ObjectRecord | None = None
    if register:
        service = CatalogueService()
        existing = service.get_object(object_id)
        if existing:
            payload = apply_publish_result(existing, result)
            record = service.register_object(ObjectRecord.from_dict(payload))
        else:
            logger.warning(
                f"No catalogue entry for {object_id} — mesh published but not registered"
            )

    return result, record


def process_demo_meshes() -> list[MeshPublishResult]:
    """
    Generate smooth demo OBJ meshes for all existing catalogue entries.

    Used when Meshroom is not installed — produces valid CAD-style geometry
    so the dashboard viewer can be tested end-to-end.

    Returns:
        List of publish results, one per catalogue object
    """
    from src.photogrammetry.meshroom import generate_demo_mesh

    service = CatalogueService()
    results: list[MeshPublishResult] = []

    for entry in service.load_all_objects():
        object_id = entry["object_id"]
        class_name = entry.get("class_name", "pottery")
        result = generate_demo_mesh(object_id, class_name)
        payload = apply_publish_result(entry, result)
        service.register_object(ObjectRecord.from_dict(payload))
        results.append(result)

    return results
