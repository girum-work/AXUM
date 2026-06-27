# scripts/seed_demo_catalogue.py
"""
Seed demo catalogue entries for the interactive 3D catalogue UI.

Creates three objects with one fragment group (pottery shards 001–003 are
NOT grouped in demo — OBJ-001/002 share group; OBJ-003 is inscription).
Run: python scripts/seed_demo_catalogue.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CATALOGUE_DIR, FRAGMENT_GROUPS_JSON
from src.catalogue.records import ObjectRecord
from src.catalogue.service import CatalogueService


def main() -> None:
    """Write demo catalogue JSON + PDF for catalogue viewer testing."""
    # Fresh seed — remove prior demo artefacts so groups rebuild cleanly
    for old in CATALOGUE_DIR.glob("AXUM-OBJ-*.json"):
        old.unlink()
    if FRAGMENT_GROUPS_JSON.exists():
        FRAGMENT_GROUPS_JSON.unlink()

    service = CatalogueService()

    hist_pottery = [0.12, 0.18, 0.14, 0.09, 0.06, 0.08, 0.09, 0.05,
                    0.04, 0.04, 0.04, 0.03, 0.02, 0.02, 0.01, 0.04]

    demos = [
        ObjectRecord(
            object_id="AXUM-OBJ-001",
            sequence_number=1,
            class_name="pottery",
            class_confidence=0.91,
            density_g_cm3=1.84,
            colour_histogram=hist_pottery,
            uv_signature=0.32,
            crack_depth_mm=0.8,
            crack_severity=0.25,
            fragility_years=87.0,
            salt_stage="None detected",
            interventions=["RFID implanted"],
            photo_count=36,
            mesh_duration=252.0,
        ),
        ObjectRecord(
            object_id="AXUM-OBJ-002",
            sequence_number=2,
            class_name="pottery",
            class_confidence=0.88,
            density_g_cm3=1.79,
            colour_histogram=[h * 0.96 + 0.01 for h in hist_pottery],
            uv_signature=0.30,
            inscription_text="ሰ[MISSING]ም",
            translation_en="Peace (partial inscription on shard)",
            ocr_confidence=0.78,
            crack_depth_mm=1.2,
            crack_severity=0.40,
            fragility_years=45.0,
            salt_stage="Trace fluorescence",
            interventions=["RFID implanted"],
            photo_count=36,
            mesh_duration=248.0,
        ),
        ObjectRecord(
            object_id="AXUM-OBJ-003",
            sequence_number=3,
            class_name="pottery",
            class_confidence=0.87,
            density_g_cm3=1.81,
            colour_histogram=[h * 0.97 for h in hist_pottery],
            uv_signature=0.31,
            inscription_text="[MISSING]ም",
            ocr_confidence=0.55,
            crack_depth_mm=0.5,
            fragility_years=95.0,
            salt_stage="Trace fluorescence",
            interventions=["RFID implanted"],
            photo_count=36,
            mesh_duration=261.0,
        ),
        ObjectRecord(
            object_id="AXUM-OBJ-004",
            sequence_number=4,
            class_name="coin",
            class_confidence=0.94,
            density_g_cm3=8.71,
            inscription_text="ዓጼ [MISSING]ዛ[MISSING]",
            translation_en="Emperor Ezana",
            ocr_confidence=0.82,
            fragility_years=340.0,
            salt_stage="Historical residue",
            interventions=["RFID implanted"],
            photo_count=36,
            is_new_discovery=True,
        ),
    ]

    for rec in demos:
        service.register_object(rec)

    print(f"Seeded {len(demos)} catalogue entries -> {service.output_dir}")
    print(f"Fragment groups: {len(service.grouper.groups)}")
    for g in service.grouper.groups:
        print(f"  {g.group_id}: {g.confirmed_members} (+{g.possible_members} possible)")


if __name__ == "__main__":
    main()
