# scripts/run_meshroom.py
"""
Run Meshroom photogrammetry for one catalogue object and publish the mesh.

Usage:
    python scripts/run_meshroom.py AXUM-OBJ-001
    python scripts/run_meshroom.py AXUM-OBJ-001 --skip-meshroom  # publish existing export only
    python scripts/run_meshroom.py AXUM-OBJ-001 --export-dir path/to/meshroom/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipeline.mesh_stage import process_object_mesh


def main() -> None:
    """CLI entry point for Meshroom photogrammetry on one object."""
    parser = argparse.ArgumentParser(description="Run Meshroom for one AXUM object")
    parser.add_argument("object_id", help="Catalogue ID e.g. AXUM-OBJ-001")
    parser.add_argument(
        "--skip-meshroom",
        action="store_true",
        help="Publish from existing cache/export without running Meshroom",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Explicit Meshroom export directory to publish",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Publish mesh files only — do not update catalogue JSON",
    )
    args = parser.parse_args()

    result, record = process_object_mesh(
        args.object_id,
        skip_meshroom=args.skip_meshroom,
        export_dir=args.export_dir,
        register=not args.no_register,
    )

    print(f"Published: {result.mesh_path}")
    print(f"Vertices:  {result.vertex_count}")
    print(f"Faces:     {result.face_count}")
    print(f"Duration:  {result.duration_sec:.1f}s")
    if record:
        print(f"Catalogue: {record.object_id} updated")
    for warning in result.warnings:
        logger.warning(warning)


if __name__ == "__main__":
    main()
