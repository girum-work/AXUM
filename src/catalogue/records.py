# src/catalogue/records.py
"""
AXUM ROVER — Catalogue Record Types
====================================
Shared dataclasses for artefact scan results. Used by the catalogue
generator, fragment grouper, Flask API, and (future) main pipeline.

Author: Axum Rover Team
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format for catalogue records."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ObjectRecord:
    """
    One scanned artefact — the unit stored in the heritage catalogue.

    Fields are populated incrementally as the pipeline runs (classify,
    weigh, OCR, 3D mesh, fragment grouping). Optional fields default
    to None/empty so partial records are valid during scanning.

    Attributes:
        object_id:          Unique ID e.g. AXUM-OBJ-001
        sequence_number:    Mission order (1, 2, 3…)
        class_name:         pottery | stone_carving | coin | inscription_fragment | other
        class_confidence:   Classifier confidence 0–1
        class_source:       yolo | mobilenet | manual
        density_g_cm3:      Load-cell mass / estimated volume
        width_mm, height_mm, depth_mm: Bounding dimensions
        inscription_text:   Ge'ez OCR output (may contain [MISSING])
        translation_en:     English translation
        ocr_confidence:     OCR confidence 0–1
        crack_depth_mm:     Deepest crack from photometric stereo
        crack_severity:     Normalised severity 0–1
        salt_stage:         Human-readable salt risk label
        fragility_years:    Years-remaining estimate
        mesh_path:          Path to Meshroom .obj file
        photo_paths:        List of turntable image paths
        photo_count:        Number of scan photos captured
        mesh_duration:      Seconds to build 3D model
        colour_histogram:   16-bin RGB histogram for material matching
        uv_signature:       Mean UV fluorescence intensity 0–1
        interventions:      Physical actions taken (RFID, consolidant, etc.)
        group_id:           Fragment group ID if grouped (AXUM-GRP-001)
        group_conf:         Reconstruction confidence for this object's group
        group_role:         confirmed | possible | ungrouped
        match_scores:       Dict of other_object_id → pairwise match score
        is_new_discovery:   True if inscription not in known database
        db_match_id:        Matched inscription database entry
        timestamp:          ISO UTC scan completion time
        errors:             Non-fatal pipeline warnings
    """
    object_id:          str
    sequence_number:    int = 0
    class_name:         str = "other"
    class_confidence:   float = 0.0
    class_source:       str = ""
    density_g_cm3:      Optional[float] = None
    width_mm:           float = 0.0
    height_mm:          float = 0.0
    depth_mm:           float = 0.0
    inscription_text:   str = ""
    translation_en:     str = ""
    ocr_confidence:     float = 0.0
    crack_depth_mm:     float = 0.0
    crack_severity:     float = 0.0
    salt_stage:         str = ""
    fragility_years:    Optional[float] = None
    mesh_path:          str = ""
    photo_paths:        list[str] = field(default_factory=list)
    photo_count:        int = 0
    mesh_duration:      float = 0.0
    colour_histogram:   list[float] = field(default_factory=list)
    uv_signature:       Optional[float] = None
    interventions:      list[str] = field(default_factory=list)
    group_id:           Optional[str] = None
    group_conf:         float = 0.0
    group_role:         str = "ungrouped"
    match_scores:       dict[str, float] = field(default_factory=dict)
    is_new_discovery:   bool = False
    db_match_id:        str = ""
    timestamp:          str = field(default_factory=_utc_now_iso)
    errors:             list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON catalogue files and Flask API responses."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectRecord":
        """Reconstruct ObjectRecord from saved catalogue JSON."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
