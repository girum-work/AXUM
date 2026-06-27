# scripts/register_mesh.py
"""
Register an existing Meshroom export folder with the AXUM catalogue.

Use this after a manual Meshroom run, or to import a third-party OBJ export.

Usage:
    python scripts/register_mesh.py AXUM-OBJ-001 path/to/meshroom/output
    python scripts/register_mesh.py --sync   # back-fill mesh_path from scans/meshes/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.catalogue.mesh_registry import sync_catalogue_mesh_paths
from src.pipeline.mesh_stage import process_object_mesh


def main() -> None:
    """CLI entry point for registering a Meshroom export with the catalogue."""
    parser = argparse.ArgumentParser(description="Register Meshroom export with catalogue")
    parser.add_argument("object_id", nargs="?", help="Catalogue ID e.g. AXUM-OBJ-001")
    parser.add_argument("export_dir", nargs="?", type=Path, help="Meshroom output folder")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Back-fill mesh_path for all published meshes under scans/meshes/",
    )
    args = parser.parse_args()

    if args.sync:
        count = sync_catalogue_mesh_paths()
        print(f"Synced {count} catalogue entries")
        return

    if not args.object_id or not args.export_dir:
        parser.error("object_id and export_dir are required unless --sync is used")

    result, record = process_object_mesh(
        args.object_id,
        skip_meshroom=True,
        export_dir=args.export_dir,
        register=True,
    )
    print(f"Registered {args.object_id} → {result.mesh_path}")
    if record:
        logger.info(f"Dashboard URL: /models/{args.object_id}/{Path(result.mesh_path).name}")


if __name__ == "__main__":
    main()
