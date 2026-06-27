"""
AXUM ROVER — Mission Pipeline Package
======================================
WHAT:  Orchestrates scan stages from pick-up through catalogue registration.
WHY:   Ties hardware, AI modules, and dashboard into one mission loop.

Available now (mesh_stage.py):
    process_object_mesh(object_id)  — run Meshroom / publish OBJ
    process_demo_meshes()           — procedural demo OBJ for all catalogue entries

Coming soon (main_pipeline.py — NOT YET ON DISK):
    MissionPipeline, MissionState, SharedState — full autonomous mission

Lazy imports via __getattr__ so mesh stage works before main_pipeline exists.
"""

__all__ = [
    "process_object_mesh",
    "process_demo_meshes",
]


def __getattr__(name: str):
    """Lazy import — avoids loading main_pipeline until it is implemented."""
    if name in ("process_object_mesh", "process_demo_meshes"):
        from src.pipeline import mesh_stage
        return getattr(mesh_stage, name)
    if name in (
        "MissionPipeline", "MissionState", "ObjectRecord",
        "MissionRecord", "SharedState", "state",
    ):
        from src.pipeline import main_pipeline
        return getattr(main_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
