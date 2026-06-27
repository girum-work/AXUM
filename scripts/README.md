# AXUM ROVER — Documentation Index

Start here when joining the project or handing off work.

## For a new developer (read in this order)

1. **[SETUP.md](SETUP.md)** — Install Python, venv, dependencies, datasets, and smoke tests on a fresh PC.
2. **[TECHNICAL_HANDOFF.md](TECHNICAL_HANDOFF.md)** — Big picture: architecture, what's done, what's missing, priority next steps.
3. **[CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md)** — Every file and block explained; when to run each script.
4. **[../.cursorrules](../.cursorrules)** — Architecture contract and coding rules (mandatory before changing design).

## Quick commands after setup

```powershell
cd C:\Users\<you>\Documents\Programming\Projects\AXUM
.\venv\Scripts\activate
python verify_install.py
python scripts/seed_demo_catalogue.py
python scripts/generate_demo_meshes.py
python src/dashboard/server.py
```

## Critical gaps (as of June 2026)

| Missing | Priority |
|---------|----------|
| `src/arm/controller.py` | P0 — hardware serial layer |
| `src/pipeline/main_pipeline.py` | P0 — mission orchestrator |
| Trained weights in `models/` | P0 — run `scripts/train_ocr.py` first |
| `src/imaging/photometric_stereo.py`, `multispectral.py` | P2 |

## Document map

| File | Purpose |
|------|---------|
| `SETUP.md` | Onboarding install guide |
| `TECHNICAL_HANDOFF.md` | Handoff + priorities for teammate |
| `CODEBASE_WALKTHROUGH.md` | Block-by-block code reference |
| `../AXUM_CLAUDE_BRIEFING.md` | Full context for external AI |
| `../config.py` | All constants (with module docstring) |
| `../.cursorrules` | Non-negotiable architecture rules |
