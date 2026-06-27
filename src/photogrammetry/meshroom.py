# src/photogrammetry/meshroom.py
"""
AXUM ROVER — Meshroom Photogrammetry Pipeline
==============================================
Turns turntable photo sets into dashboard-ready 3D meshes.

End-to-end flow:
    1. Turntable captures land in ``scans/photos/<object_id>/``
    2. Meshroom batch runs (optional — skipped if export already exists)
    3. Best ``.obj`` export is copied to ``scans/meshes/<object_id>/model.obj``
       together with ``.mtl`` and texture images
    4. Catalogue JSON ``mesh_path`` is updated for the dashboard viewer

The dashboard loads meshes via ``/models/<object_id>/model.obj`` with
MTL/textures served from the same folder.

Author: Axum Rover Team
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    MESHROOM_CACHE_DIR,
    MESHROOM_OBJ_PRIORITY,
    MESHROOM_PATH,
    MESHROOM_TIMEOUT_SEC,
    MESH_DIR,
    MESH_PUBLISH_MTL,
    MESH_PUBLISH_NAME,
    MESH_TEXTURE_EXTENSIONS,
    ROOT_DIR,
    SCAN_PHOTOS_DIR,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class MeshPublishResult:
    """
    Result of publishing one Meshroom export for dashboard viewing.

    Attributes:
        object_id:      Catalogue ID e.g. AXUM-OBJ-001
        mesh_path:      Path stored in catalogue JSON (relative to project root)
        obj_file:       Absolute path to published OBJ
        mtl_file:       Absolute path to published MTL, if any
        texture_files:  Copied texture images beside the OBJ
        vertex_count:   Number of ``v`` vertices in OBJ (0 if unknown)
        face_count:     Number of ``f`` faces in OBJ (0 if unknown)
        duration_sec:   Seconds spent publishing / reconstructing
        source_obj:     Original Meshroom OBJ before normalisation
        warnings:       Non-fatal issues (missing MTL, renamed textures, etc.)
    """
    object_id:      str
    mesh_path:      str
    obj_file:       Path
    mtl_file:       Path | None = None
    texture_files:  list[Path] = field(default_factory=list)
    vertex_count:   int = 0
    face_count:     int = 0
    duration_sec:   float = 0.0
    source_obj:     Path | None = None
    warnings:       list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — PATH HELPERS
# ═══════════════════════════════════════════════════════════════

def object_photo_dir(object_id: str) -> Path:
    """
    Directory where turntable photos for one object are stored.

    Args:
        object_id: Catalogue object ID

    Returns:
        ``scans/photos/<object_id>/``
    """
    return SCAN_PHOTOS_DIR / object_id


def object_mesh_dir(object_id: str) -> Path:
    """
    Canonical published mesh directory for one object.

    Args:
        object_id: Catalogue object ID

    Returns:
        ``scans/meshes/<object_id>/``
    """
    return MESH_DIR / object_id


def object_meshroom_cache(object_id: str) -> Path:
    """
    Meshroom working/output directory for one object.

    Args:
        object_id: Catalogue object ID

    Returns:
        ``scans/meshroom_cache/<object_id>/``
    """
    return MESHROOM_CACHE_DIR / object_id


def resolve_mesh_path(mesh_path: str | Path) -> Path:
    """
    Resolve a catalogue ``mesh_path`` string to an absolute filesystem path.

    Args:
        mesh_path: Relative or absolute path from catalogue JSON

    Returns:
        Absolute ``Path`` under project root when relative
    """
    path = Path(mesh_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def mesh_path_for_catalogue(obj_file: Path) -> str:
    """
    Convert an absolute OBJ path to the relative form stored in catalogue JSON.

    Args:
        obj_file: Absolute published OBJ path

    Returns:
        Forward-slash relative path from project root
    """
    try:
        rel = obj_file.relative_to(ROOT_DIR)
    except ValueError:
        rel = obj_file
    return rel.as_posix()


def mesh_public_url(object_id: str, mesh_path: str | Path) -> str:
    """
    Build the dashboard HTTP URL for a published mesh OBJ.

    Args:
        object_id: Catalogue object ID
        mesh_path: Stored catalogue mesh path

    Returns:
        URL path e.g. ``/models/AXUM-OBJ-001/model.obj``
    """
    filename = Path(mesh_path).name
    return f"/models/{object_id}/{filename}"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — MESHROOM DISCOVERY + VALIDATION
# ═══════════════════════════════════════════════════════════════

def find_meshroom_obj(export_dir: Path) -> Path | None:
    """
    Locate the best OBJ file inside a Meshroom output tree.

    Prefers textured exports (``texturedMesh.obj``) then falls back to the
    largest ``.obj`` file found recursively.

    Args:
        export_dir: Meshroom output root directory

    Returns:
        Path to chosen OBJ, or ``None`` if no OBJ exists
    """
    export_dir = Path(export_dir)
    if not export_dir.exists():
        return None

    candidates = sorted(export_dir.rglob("*.obj"))
    if not candidates:
        return None

    priority = {name.lower(): idx for idx, name in enumerate(MESHROOM_OBJ_PRIORITY)}
    ranked = sorted(
        candidates,
        key=lambda p: (
            priority.get(p.name.lower(), len(MESHROOM_OBJ_PRIORITY)),
            -p.stat().st_size,
        ),
    )
    return ranked[0]


def _parse_obj_stats(obj_path: Path) -> tuple[int, int]:
    """
    Count vertices and faces in an OBJ without external dependencies.

    Args:
        obj_path: OBJ file path

    Returns:
        Tuple of (vertex_count, face_count)
    """
    vertices = 0
    faces = 0
    with open(obj_path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return vertices, faces


def validate_mesh(obj_path: Path) -> tuple[bool, str]:
    """
    Verify an OBJ is readable and contains geometry.

    Args:
        obj_path: Published or source OBJ path

    Returns:
        ``(ok, message)`` — message explains failure when ``ok`` is False
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        return False, f"OBJ not found: {obj_path}"
    if obj_path.stat().st_size < 32:
        return False, f"OBJ too small: {obj_path}"

    vertices, faces = _parse_obj_stats(obj_path)
    if vertices < 4:
        return False, f"OBJ has too few vertices ({vertices}): {obj_path}"
    if faces < 4:
        return False, f"OBJ has too few faces ({faces}): {obj_path}"
    return True, f"OK — {vertices} vertices, {faces} faces"


def _related_textures(source_obj: Path) -> list[Path]:
    """
    Find texture images referenced by an OBJ/MTL pair in the same folder.

    Args:
        source_obj: Source OBJ path

    Returns:
        List of existing texture file paths to copy
    """
    folder = source_obj.parent
    textures: list[Path] = []
    seen: set[str] = set()

    mtl_name = None
    with open(source_obj, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.lower().startswith("mtllib "):
                mtl_name = line.split(maxsplit=1)[1].strip()
                break

    candidates: set[str] = set()
    if mtl_name:
        mtl_path = folder / mtl_name
        if mtl_path.exists():
            with open(mtl_path, encoding="utf-8", errors="ignore") as mtl:
                for line in mtl:
                    for token in line.strip().split():
                        if token.lower().endswith(MESH_TEXTURE_EXTENSIONS):
                            candidates.add(token)

    for ext in MESH_TEXTURE_EXTENSIONS:
        for path in folder.glob(f"*{ext}"):
            candidates.add(path.name)

    for name in sorted(candidates):
        path = folder / name
        if path.exists() and path.is_file() and name not in seen:
            seen.add(name)
            textures.append(path)
    return textures


def _rewrite_mtl_texture_refs(mtl_path: Path, name_map: dict[str, str]) -> None:
    """
    Update texture filenames inside a published MTL after copying assets.

    Args:
        mtl_path: Destination MTL path
        name_map: Old basename → new basename mapping
    """
    if not mtl_path.exists() or not name_map:
        return
    text = mtl_path.read_text(encoding="utf-8", errors="ignore")
    for old, new in name_map.items():
        text = re.sub(re.escape(old), new, text)
    mtl_path.write_text(text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PUBLISH EXPORT → DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════

def publish_mesh_export(
    object_id: str,
    export_dir: Path,
    *,
    source_obj: Path | None = None,
) -> MeshPublishResult:
    """
    Copy a Meshroom OBJ export into the canonical dashboard mesh folder.

    Normalises filenames to ``model.obj`` / ``model.mtl`` so the viewer always
    knows which files to request. Updates are written to
    ``scans/meshes/<object_id>/``.

    Args:
        object_id:  Catalogue object ID
        export_dir: Meshroom output directory (used when ``source_obj`` omitted)
        source_obj: Explicit OBJ path; when None, ``find_meshroom_obj`` is used

    Returns:
        ``MeshPublishResult`` with published paths and mesh statistics

    Raises:
        FileNotFoundError: When no OBJ can be located
        ValueError: When published OBJ fails validation
    """
    t0 = time.perf_counter()
    warnings: list[str] = []

    export_dir = Path(export_dir)
    src_obj = Path(source_obj) if source_obj else find_meshroom_obj(export_dir)
    if src_obj is None or not src_obj.exists():
        raise FileNotFoundError(
            f"No OBJ export found for {object_id} under {export_dir}"
        )

    ok, msg = validate_mesh(src_obj)
    if not ok:
        raise ValueError(f"Invalid OBJ for {object_id}: {msg}")

    dest_dir = object_mesh_dir(object_id)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_obj = dest_dir / MESH_PUBLISH_NAME
    shutil.copy2(src_obj, dest_obj)

    dest_mtl: Path | None = None
    src_mtl = src_obj.with_suffix(".mtl")
    if not src_mtl.exists():
        with open(src_obj, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.lower().startswith("mtllib "):
                    src_mtl = src_obj.parent / line.split(maxsplit=1)[1].strip()
                    break

    texture_files: list[Path] = []
    tex_name_map: dict[str, str] = {}
    for idx, tex in enumerate(_related_textures(src_obj), start=1):
        dest_name = f"texture_{idx:02d}{tex.suffix.lower()}"
        dest_tex = dest_dir / dest_name
        shutil.copy2(tex, dest_tex)
        texture_files.append(dest_tex)
        tex_name_map[tex.name] = dest_name

    if src_mtl.exists():
        dest_mtl = dest_dir / MESH_PUBLISH_MTL
        shutil.copy2(src_mtl, dest_mtl)
        _rewrite_mtl_texture_refs(dest_mtl, tex_name_map)

        obj_text = dest_obj.read_text(encoding="utf-8", errors="ignore")
        obj_text = re.sub(
            r"(?m)^mtllib\s+.*$",
            f"mtllib {MESH_PUBLISH_MTL}",
            obj_text,
            count=1,
        )
        dest_obj.write_text(obj_text, encoding="utf-8")
    else:
        warnings.append("No MTL found — dashboard will use neutral CAD shading")

    vertices, faces = _parse_obj_stats(dest_obj)
    rel_path = mesh_path_for_catalogue(dest_obj)
    duration = time.perf_counter() - t0

    logger.info(
        f"Published mesh for {object_id}: {rel_path} "
        f"({vertices} v / {faces} f, {duration:.1f}s)"
    )

    return MeshPublishResult(
        object_id=object_id,
        mesh_path=rel_path,
        obj_file=dest_obj,
        mtl_file=dest_mtl,
        texture_files=texture_files,
        vertex_count=vertices,
        face_count=faces,
        duration_sec=duration,
        source_obj=src_obj,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MESHROOM BATCH RUNNER
# ═══════════════════════════════════════════════════════════════

def _meshroom_cli_candidates() -> list[Path]:
    """
    Build a list of possible Meshroom batch executables on this machine.

    Returns:
        Existing executable paths, most preferred first
    """
    root = Path(MESHROOM_PATH)
    candidates = [
        root,
        root.parent / "meshroom_photogrammetry.exe",
        root.parent / "aliceVision_meshroom_photogrammetry.exe",
    ]
    return [p for p in candidates if p.exists()]


def run_meshroom_batch(
    photo_dir: Path,
    output_dir: Path,
    *,
    timeout_sec: int = MESHROOM_TIMEOUT_SEC,
) -> Path:
    """
    Run Meshroom photogrammetry on a folder of turntable images.

    Requires Meshroom to be installed at ``MESHROOM_PATH``. If Meshroom is
    not installed, raises ``FileNotFoundError`` so callers can degrade
    gracefully during development.

    Args:
        photo_dir:   Directory containing ``.jpg`` / ``.png`` photos
        output_dir:  Meshroom output/cache directory
        timeout_sec: Subprocess timeout in seconds

    Returns:
        ``output_dir`` when reconstruction completes

    Raises:
        FileNotFoundError: Meshroom executable or photos missing
        RuntimeError: Meshroom exited with non-zero status
        TimeoutError: Reconstruction exceeded ``timeout_sec``
    """
    photo_dir = Path(photo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(photo_dir.glob("*.jpg")) + list(photo_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No photos in {photo_dir}")

    cli = _meshroom_cli_candidates()
    if not cli:
        raise FileNotFoundError(
            f"Meshroom not found — set MESHROOM_PATH in config.py "
            f"(currently {MESHROOM_PATH})"
        )

    exe = cli[0]
    if exe.name.lower().endswith("meshroom.exe"):
        cmd = [
            str(exe),
            "--batch",
            f"input={photo_dir.resolve()}",
            f"output={output_dir.resolve()}",
        ]
    else:
        cmd = [
            str(exe),
            "--input", str(photo_dir.resolve()),
            "--output", str(output_dir.resolve()),
        ]

    logger.info(f"Meshroom: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(
            f"Meshroom failed (code {proc.returncode}) after {elapsed:.0f}s:\n{tail}"
        )

    logger.info(f"Meshroom finished in {elapsed:.0f}s → {output_dir}")
    return output_dir


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — END-TO-END PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_photogrammetry(
    object_id: str,
    *,
    skip_meshroom: bool = False,
    export_dir: Path | None = None,
) -> MeshPublishResult:
    """
    Full photogrammetry stage for one catalogue object.

    When ``skip_meshroom`` is False and photos exist, runs Meshroom batch.
    When ``export_dir`` is provided, publishes from that folder directly
    (useful after manual Meshroom runs).

    Args:
        object_id:     Catalogue object ID
        skip_meshroom: If True, only publish from existing export/cache
        export_dir:    Optional explicit Meshroom export directory

    Returns:
        ``MeshPublishResult`` ready for catalogue registration
    """
    t0 = time.perf_counter()
    cache_dir = export_dir or object_meshroom_cache(object_id)
    photos = object_photo_dir(object_id)

    if export_dir is None and not skip_meshroom and photos.exists():
        try:
            run_meshroom_batch(photos, cache_dir)
        except FileNotFoundError as exc:
            logger.warning(f"Meshroom unavailable for {object_id}: {exc}")
        except (RuntimeError, TimeoutError) as exc:
            logger.error(f"Meshroom failed for {object_id}: {exc}")
            raise

    result = publish_mesh_export(object_id, cache_dir)
    result.duration_sec = time.perf_counter() - t0
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — DEMO MESH GENERATOR (no Meshroom required)
# ═══════════════════════════════════════════════════════════════

def _write_lathe_obj(
    path: Path,
    profile: list[tuple[float, float]],
    segments: int = 96,
) -> None:
    """
    Write a smooth lathe-surface OBJ with vertex normals.

    Args:
        path: Output OBJ path
        profile: List of (radius, height) points for ``LatheGeometry``-style surface
        segments: Angular subdivisions around Y axis
    """
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []

    for r, y in profile:
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            verts.append((r * math.cos(theta), y, r * math.sin(theta)))
            nr = math.cos(theta)
            nz = math.sin(theta)
            norms.append((nr, 0.0, nz))

    faces: list[tuple[int, int, int]] = []
    rings = len(profile)
    for ring in range(rings - 1):
        for i in range(segments):
            j = (i + 1) % segments
            a = ring * segments + i + 1
            b = ring * segments + j + 1
            c = (ring + 1) * segments + j + 1
            d = (ring + 1) * segments + i + 1
            faces.append((a, b, c))
            faces.append((a, c, d))

    lines = ["# AXUM demo mesh", f"# vertices {len(verts)}", f"# faces {len(faces)}"]
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for nx, ny, nz in norms:
        lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
    for a, b, c in faces:
        lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_demo_mesh(object_id: str, class_name: str = "pottery") -> MeshPublishResult:
    """
    Create a valid smooth OBJ for dashboard testing without Meshroom.

    Args:
        object_id:  Catalogue object ID
        class_name: Artefact class — selects vase / slab / coin shape

    Returns:
        Published ``MeshPublishResult`` identical to a real Meshroom export
    """
    cache = object_meshroom_cache(object_id)
    cache.mkdir(parents=True, exist_ok=True)
    src = cache / "demo_export.obj"

    if class_name == "coin":
        profile = [(0.58, 0.0), (0.58, 0.04), (0.58, 0.0), (0.58, -0.04), (0.58, 0.0)]
    elif class_name in ("stone_carving", "inscription_fragment"):
        profile = [
            (0.05 + 0.55 * abs(math.sin(t * math.pi)), (t - 0.5) * 0.25)
            for t in [i / 16 for i in range(17)]
        ]
    else:
        profile = [
            (0.18 + 0.34 * math.pow(math.sin(t * math.pi), 0.85) + 0.06 * (1 - t),
             (t - 0.42) * 1.35)
            for t in [i / 32 for i in range(33)]
        ]

    _write_lathe_obj(src, profile)
    return publish_mesh_export(object_id, cache, source_obj=src)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — STANDALONE SYNTHETIC TEST
# ═══════════════════════════════════════════════════════════════

def _synthetic_test() -> None:
    """Generate and validate a demo mesh without hardware or Meshroom."""
    result = generate_demo_mesh("AXUM-OBJ-TEST", "pottery")
    ok, msg = validate_mesh(result.obj_file)
    assert ok, msg
    assert result.vertex_count > 100
    logger.info(f"Synthetic mesh OK: {result.mesh_path} — {msg}")


if __name__ == "__main__":
    _synthetic_test()
