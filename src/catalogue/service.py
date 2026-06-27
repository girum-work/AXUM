# src/catalogue/service.py
"""
AXUM ROVER — Catalogue Service
===============================
High-level API for registering scanned objects: runs fragment grouping,
writes JSON catalogue entries, and updates the PDF catalogue.

Called by main_pipeline (when built) and demo/seed scripts.

Author: Axum Rover Team
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CATALOGUE_DIR
from src.catalogue.records import ObjectRecord
from src.catalogue.generator import CatalogueGenerator
from src.catalogue.mesh_registry import enrich_catalogue_entry
from src.analysis.fragment_grouper import FragmentGrouper


class CatalogueService:
    """
    Orchestrates catalogue JSON, PDF, and fragment grouping for one mission.

    Attributes:
        output_dir: Directory for per-object JSON and axum_catalogue.pdf
        generator:  PDF/JSON catalogue generator
        grouper:    Running fragment group database
    """

    def __init__(self, output_dir: Path = CATALOGUE_DIR):
        """
        Initialise catalogue service with output directory.

        Args:
            output_dir: Where catalogue JSON and PDF are written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = CatalogueGenerator(self.output_dir)
        self.grouper = FragmentGrouper()

    def register_object(
        self,
        record: Union[ObjectRecord, dict],
    ) -> ObjectRecord:
        """
        Register one scanned object: group fragments, save JSON, update PDF.

        Args:
            record: Complete or partial ObjectRecord from pipeline

        Returns:
            Updated ObjectRecord with group_id and match_scores filled in
        """
        if isinstance(record, dict):
            record = ObjectRecord.from_dict(record)

        matches = self.grouper.register_object(record)
        updated = self.grouper.objects[record.object_id]

        # Re-save every group member — earlier fragments get group_id retroactively
        ids_to_save = {updated.object_id}
        if updated.group_id:
            group = self.grouper.get_group(updated.group_id)
            if group:
                ids_to_save.update(group.all_members())

        for oid in ids_to_save:
            member = self.grouper.objects.get(oid)
            if not member:
                continue
            json_path = self.output_dir / f"{oid}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(member.to_dict(), f, indent=2, ensure_ascii=False)

        self.generator.add_entry(updated)

        if matches:
            logger.info(
                f"Catalogue: {updated.object_id} linked to "
                f"{updated.group_id} ({len(matches)} matches)"
            )
        else:
            logger.info(f"Catalogue: {updated.object_id} registered (ungrouped)")

        return updated

    def load_all_objects(self) -> list[dict]:
        """
        Load all AXUM-OBJ-*.json files from catalogue directory.

        Returns:
            List of object dicts sorted by sequence_number, enriched with
            ``mesh_ready`` and ``mesh_url`` for the dashboard viewer
        """
        objects = []
        for path in sorted(self.output_dir.glob("AXUM-OBJ-*.json")):
            with open(path, encoding="utf-8") as f:
                objects.append(enrich_catalogue_entry(json.load(f)))
        objects.sort(key=lambda o: o.get("sequence_number", 0))
        return objects

    def get_object(self, object_id: str) -> dict | None:
        """Load single catalogue entry by object ID with mesh metadata."""
        path = self.output_dir / f"{object_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return enrich_catalogue_entry(json.load(f))

    def update_mesh(
        self,
        object_id: str,
        mesh_path: str,
        mesh_duration: float = 0.0,
    ) -> ObjectRecord | None:
        """
        Update an existing catalogue entry with a published mesh path.

        Args:
            object_id:     Catalogue object ID
            mesh_path:     Relative path to published OBJ
            mesh_duration: Reconstruction time in seconds

        Returns:
            Updated ObjectRecord, or None if object not found
        """
        existing = self.get_object(object_id)
        if not existing:
            logger.warning(f"Cannot update mesh — unknown object {object_id}")
            return None
        existing["mesh_path"] = mesh_path
        if mesh_duration > 0:
            existing["mesh_duration"] = mesh_duration
        return self.register_object(ObjectRecord.from_dict(existing))
