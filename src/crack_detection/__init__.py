"""
AXUM ROVER — Crack Detection Package
=====================================
WHAT:  Find surface cracks on artefact photos using pure OpenCV (no ML).
WHY:   Crack severity feeds treatment advisor and fragility estimates.

Standalone test (synthetic image, no hardware):
    python src/crack_detection/detector.py

Main API: run_crack_detection() in detector.py
Config:   CANNY_T1, MIN_CRACK_AREA, CRACK_SEVERITY_THRESHOLD in config.py
"""

# Main implementation: src/crack_detection/detector.py
