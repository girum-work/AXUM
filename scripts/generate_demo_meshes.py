# scripts/generate_demo_meshes.py
"""
Generate smooth demo OBJ meshes for all catalogue entries (no Meshroom required).

Creates valid Meshroom-style exports under scans/meshes/<object_id>/model.obj
and updates catalogue JSON so the dashboard loads real OBJ files instead of
procedural Three.js placeholders.

Usage:
    python scripts/generate_demo_meshes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.mesh_stage import process_demo_meshes


def main() -> None:
    """Generate and register demo meshes for every catalogue object."""
    results = process_demo_meshes()
    print(f"Generated {len(results)} demo meshes:")
    for result in results:
        print(
            f"  {result.object_id}: {result.mesh_path} "
            f"({result.vertex_count} v / {result.face_count} f)"
        )
    print("\nRestart dashboard and open /catalogue — meshes load from /models/...")


if __name__ == "__main__":
    main()
