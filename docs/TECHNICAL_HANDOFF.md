# AXUM Rover — Technical Handoff Guide

**Audience:** You (team lead) explaining the codebase to your software teammate  
**Their onboarding path:** Read this → `SETUP.md` → `CODEBASE_WALKTHROUGH.md` → `.cursorrules`

---

## 1. Elevator pitch (30 seconds)

AXUM is a **WRO 2026 Future Innovators** robot that picks up Ethiopian cultural artefacts, scans them (camera, UV, mechanical sensors), runs **local AI on a laptop** (Ge'ez OCR, crack detection, classification, conservation advice), optionally performs physical conservation actions, and outputs a **museum PDF catalogue + 3D meshes + live dashboard**.

Software runs on **Windows, CPU-only, no cloud**. Hardware is controlled by an **Arduino Mega** over USB serial; the **ESP32-CAM** streams video over WiFi.

**Deadline:** National selection, mid-August 2026. **Team:** 2 people.

---

## 2. How the software is organized (mental model)

Think of four layers:

```
┌─────────────────────────────────────────────────────────┐
│  DASHBOARD (Flask + WebSocket)                          │
│  src/dashboard/server.py — what judges see in browser   │
└───────────────────────────┬─────────────────────────────┘
                            │ emit_event()
┌───────────────────────────▼─────────────────────────────┐
│  MISSION PIPELINE (NOT BUILT YET)                       │
│  src/pipeline/main_pipeline.py — orchestrates everything│
└───────────────────────────┬─────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────┐
│  ANALYSIS MODULES (mostly built)                        │
│  OCR, cracks, salt, classifier, treatment, catalogue  │
└───────────────────────────┬─────────────────────────────┘
                            │ commands
┌───────────────────────────▼─────────────────────────────┐
│  HARDWARE LAYER (NOT BUILT YET)                         │
│  src/arm/controller.py — serial to Arduino              │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** Most AI/analysis code is written. What's missing is the **glue** — hardware controller + mission state machine — that ties modules together for a live demo.

---

## 3. The one file that rules them all: `config.py`

Every path, threshold, COM port, and model filename lives in `config.py`. **No magic numbers anywhere else.**

When your teammate asks "where do I change X?" → almost always `config.py`.

Important groups:
- **Paths:** `DATA_DIR`, `MODELS_DIR`, `SCAN_PHOTOS_DIR`, `CATALOGUE_DIR`
- **Hardware:** `SERIAL_PORT`, `ESP32_CAM_URL`, `TURNTABLE_STEPS=36`
- **OCR:** `OCR_IMG_SIZE=(32,128)`, `OCR_CONFIDENCE_MIN=0.50`
- **Classifier:** `OBJ_CLASSES` (5 artefact types)
- **Treatment:** `CONSERVATION_KB_PATH`, `ARTEFACT_SUBSTRATE_MAP`

See `CODEBASE_WALKTHROUGH.md` § config.py for line-by-line explanation.

---

## 4. What each folder does

| Folder | Purpose | Status |
|--------|---------|--------|
| `src/ocr/` | Ge'ez OCR model + training + LLM restoration | ✅ Substantial |
| `src/object_detection/` | Tray detection + MobileNet + YOLO + coins | ✅ Substantial |
| `src/crack_detection/` | OpenCV crack finder | ✅ Done |
| `src/imaging/` | Salt mapper only | ⚠️ Partial (multispectral/stereo missing) |
| `src/analysis/` | Treatment advisor + fragment grouper | ✅ Done |
| `src/catalogue/` | JSON records + PDF generator + service | ✅ Done |
| `src/photogrammetry/` | Meshroom wrapper + demo meshes | ✅ Done |
| `src/pipeline/` | Mesh stage only | ⚠️ `main_pipeline.py` missing |
| `src/dashboard/` | Flask-SocketIO UI | ✅ Done |
| `src/arm/` | Hardware control | ❌ `controller.py` missing |
| `scripts/` | Training, download, verification, demos | ✅ Many scripts |
| `arduino/` | Mega + ESP32 firmware | ✅ Written, untested |
| `data/` | Datasets, KB, demo catalogue | ✅ OCR data ready |
| `models/` | Trained weights | ❌ Empty — must train |
| `scans/` | Photos + meshes | ⚠️ Demo meshes only |

---

## 5. The data flow for one artefact (target end state)

Walk your teammate through this sequence — it's what `main_pipeline.py` must implement:

1. **Detect** objects in tray → `detect_and_classify_tray()` in `object_detection/detector.py`
2. **Pick** → `ArmController` sends `POSE:PICK`, `GRIP`, `POSE:PLACE` (needs `controller.py`)
3. **Turntable scan** → 36× `ROTATE` + `PHOTO` → images land in `scans/photos/<object_id>/`
4. **Crack analysis** → `crack_detection/detector.py` → severity 0–1
5. **Salt mapping** → `imaging/salt_mapper.py` → risk level 0–4
6. **OCR** (if inscription/coin) → `ocr/pipeline.py` → Ge'ez text
7. **LLM restore** → `ocr/llm_restoration.py` → translation
8. **Treatment** → `analysis/treatment_advisor.py` → safe/unsafe protocol
9. **Fragment match** → `analysis/fragment_grouper.py` → group_id
10. **3D mesh** → `pipeline/mesh_stage.py` → Meshroom or demo OBJ
11. **Catalogue** → `catalogue/service.py` → JSON + PDF
12. **Dashboard** → `dashboard/server.py` → `emit_event()` at each step

The **`ObjectRecord`** dataclass (`catalogue/records.py`) accumulates fields as stages complete.

---

## 6. AI models — what exists vs what needs training

| Model | Architecture | Training script | Weights |
|-------|---------------|-----------------|---------|
| Ge'ez OCR | CNN + BiLSTM + CTC | `scripts/train_ocr.py` | ❌ Not trained |
| LLM restoration | Qwen2.5-1.5B (fine-tune on Colab) | `scripts/export_restoration_colab.py` | ❌ Not fine-tuned |
| Artefact classifier | MobileNetV2 | `scripts/train_classifier.py` | ❌ Not trained |
| YOLO artefacts | YOLO11n | `scripts/train_yolo11.py` | ❌ Not trained |
| YOLO coins | YOLO11n (7 classes) | `scripts/train_yolo11_coins.py` | ❌ Not trained |
| Crack detector | OpenCV only | None | N/A |

**Dataset status:**
- OCR: **ready** (~79,684 images in `data/geez_characters/`)
- Classifier: **short** (~228 accepted images; need 500+)
- Coins: **not collected** to target

**Highest ROI task:** Train OCR first — dataset is ready, code is ready, only CPU time is needed.

---

## 7. Critical missing files (blockers)

### P0 — Must build before live robot demo

#### `src/arm/controller.py`
Expected exports (already imported in `src/arm/__init__.py`):
- **`ArduinoSerial`** — wraps pyserial, `send_command("PING")` with retry
- **`ArmController`** — high-level poses: `go_home()`, `pick()`, `place()`
- **`TurntableController`** — `rotate_degrees()`, `capture_rotation_set()`
- **`CameraInterface`** — HTTP GET to ESP32 `/capture` and stream URL

All commands must match the protocol in `.cursorrules` and `arduino/axum_rover/axum_rover.ino`.

#### `src/pipeline/main_pipeline.py`
Expected exports (lazy-imported in `src/pipeline/__init__.py`):
- **`MissionPipeline`** — runs the full loop for N artefacts
- **`MissionState`** / **`SharedState`** — state dict the dashboard reads
- Integration with `emit_event()` from dashboard

**Pattern to follow:** `mesh_stage.py` shows how a pipeline stage wraps lower modules + catalogue updates.

### P1 — Planned but not blocking first integration

- `src/imaging/photometric_stereo.py`
- `src/imaging/multispectral.py`
- `src/analysis/fragility_clock.py`
- `src/sensing/acoustic_tap.py`
- Full `src/intervention/`, `src/audio/`, `src/visualization/` packages

---

## 8. What already works without hardware

Your teammate can develop and test these **today**:

| Command | What it proves |
|---------|----------------|
| `python verify_install.py` | Dependencies OK |
| `python src/imaging/salt_mapper.py` | Salt mapper synthetic test |
| `python src/crack_detection/detector.py` | Crack detection on synthetic image |
| `python src/analysis/treatment_advisor.py` | Treatment protocol from fake inputs |
| `python src/analysis/fragment_grouper.py` | Fragment matching logic |
| `python src/photogrammetry/meshroom.py` | Demo mesh generation |
| `python scripts/seed_demo_catalogue.py` | Demo catalogue JSON |
| `python scripts/generate_demo_meshes.py` | Demo OBJ files |
| `python src/dashboard/server.py` | Dashboard UI at :5000 |

---

## 9. Scripts cheat sheet (when to run what)

### One-time setup
| Script | When |
|--------|------|
| `verify_install.py` | After pip install |
| `scripts/download_datasets.py` | If OCR data missing |
| `scripts/extract_cached_dataset.py` | After HF download |
| `scripts/download_artefact_images.py` | Before classifier training |
| `scripts/build_conservation_kb.py` | Already run; KB exists |

### Training (long CPU jobs)
| Script | When | Output |
|--------|------|--------|
| `scripts/train_ocr.py` | OCR data ready | `models/geez_ocr.pth` |
| `scripts/train_classifier.py` | 500+ artefact images | `models/artefact_classifier.pth` |
| `scripts/train_yolo11.py` | Labelled detection data | `models/yolo11_artefacts.pt` |
| `scripts/train_yolo11_coins.py` | Coin images collected | `models/yolo11_coins.pt` |

### Demo / testing
| Script | When |
|--------|------|
| `scripts/seed_demo_catalogue.py` | Fresh clone, need demo data |
| `scripts/generate_demo_meshes.py` | Dashboard 3D viewer testing |
| `scripts/test_crack_spike.py` | Quick crack module check |
| `scripts/test_ocr_spike.py` | Quick OCR import check |
| `scripts/test_llm_spike.py` | LM Studio running |
| `scripts/verify_all.py` | Pre-demo full system check |

### Photogrammetry
| Script | When |
|--------|------|
| `scripts/run_meshroom.py` | Real turntable photos + Meshroom installed |
| `scripts/register_mesh.py` | Manual mesh registration |

Full block-by-block script documentation: `CODEBASE_WALKTHROUGH.md` § scripts.

---

## 10. Coding rules (non-negotiable)

From `.cursorrules` — enforce these in code review:

1. **Docstrings** on every function (WHAT, WHY, params)
2. **`loguru`** in `src/` — no `print()` except scripts
3. **Import from `config.py`** — never hardcode paths/thresholds
4. **`pathlib.Path`** for all paths
5. **Serial only via `ArduinoSerial.send_command()`**
6. **try/except** on all hardware calls with graceful fallback
7. **New imaging modules:** 7-section pattern (see `salt_mapper.py`)
8. **`__main__` synthetic test** in every new module
9. **`num_workers=0`** in all DataLoaders
10. **No GPU code** — ever

---

## 11. Recommended next steps (priority order)

Share this ordered list with your teammate:

### Week 1 — Environment + understand + first training
1. Complete `docs/SETUP.md` checklist
2. Read `.cursorrules` and this document
3. Skim `CODEBASE_WALKTHROUGH.md` (reference, not memorization)
4. Run dashboard with demo data: `seed_demo_catalogue` → `generate_demo_meshes` → `server.py`
5. Start OCR training: `python scripts/train_ocr.py` (runs overnight)

### Week 2 — Hardware layer
6. **Implement `src/arm/controller.py`**
   - Start with `ArduinoSerial.ping()` and `send_command()`
   - Add `TurntableController.rotate_degrees(10)`
   - Add `CameraInterface.capture_frame()` via requests to ESP32
7. Test against real Arduino with `PING` / `STATUS`

### Week 3 — Mission glue
8. **Implement `src/pipeline/main_pipeline.py`**
   - Start with a **single-object stub**: classify → fake scan → catalogue → dashboard events
   - Add turntable photo loop when hardware ready
   - Wire `emit_event()` at each stage

### Week 4 — Models + data
9. Download artefact images to 500+ total
10. Train classifier + YOLO
11. Integrate OCR inference into pipeline when `geez_ocr.pth` exists

### Ongoing
12. Build missing imaging modules as time allows
13. LM Studio + restoration fine-tune on Colab
14. Hardware calibration (arm poses in Arduino firmware)

---

## 12. How to explain the dashboard

`src/dashboard/server.py` is the **judge-facing UI backend**.

- Flask serves HTML templates (`dashboard.html`, `catalogue.html`)
- SocketIO pushes real-time events (`crack_detected`, `inscription_recognized`, etc.)
- **`main_pipeline.py` only needs to call `emit_event()`** — it doesn't need to know about WebSockets
- Catalogue API reads JSON from `data/catalogue/`
- 3D meshes served from `scans/meshes/<id>/model.obj`

Run: `python src/dashboard/server.py` → open `http://localhost:5000`

---

## 13. How to explain the catalogue system

Three files work together:

1. **`records.py`** — `ObjectRecord` dataclass (the schema)
2. **`generator.py`** — turns records into PDF pages (ReportLab, dark museum theme)
3. **`service.py`** — `register_object()` saves JSON, updates PDF, runs fragment grouper

Demo entries: `data/catalogue/AXUM-OBJ-001.json` through `-004.json` + `axum_catalogue.pdf`

---

## 14. How to explain OCR (the flagship AI feature)

**Problem:** Read Ge'ez (Ethiopic) inscriptions from eroded stone.

**Pipeline:**
1. `model.py` — CNN extracts features, BiLSTM reads sequence, CTC loss decodes characters
2. `pipeline.py` — preprocessing (CLAHE + adaptive threshold for stone), training loop, inference
3. `damage.py` — synthetic erosion for training robustness
4. `llm_restoration.py` — fills `[MISSING]` tokens via local LLM
5. `restoration_prompts.py` — few-shot prompts with historical examples

**Why not Transformer OCR?** CPU speed — ~0.3s/image vs 10s+ for TrOCR.

**Training:** `scripts/train_ocr.py` → saves to `models/geez_ocr.pth`

---

## 15. How to explain conservation / treatment advisor

**Problem:** Untrained field workers cause damage. Robot automates "study before you touch."

**Flow:**
- Sensor outputs (crack severity, salt risk, class, OCR confidence) → `DiagnosticInputs`
- `treatment_advisor.py` looks up substrate in `conservation_kb.json` (~28k lines, ICOMOS-based)
- Outputs ranked safe treatments, explicit **dangerous** warnings, urgency level

Run standalone test: `python src/analysis/treatment_advisor.py`

---

## 16. Known technical debt (be honest with teammate)

| Issue | Impact | Fix when |
|-------|--------|----------|
| `config.py` duplicate keys (`OCR_IMG_SIZE` defined twice) | Later values win; confusing | Before OCR training |
| `requirements.txt` duplicate/conflicting versions | pip conflicts on fresh install | SETUP.md workaround exists |
| `src/arm/controller.py` missing | Hardware imports crash | P0 |
| `main_pipeline.py` missing | No automated mission | P0 |
| `models/` empty | Inference uses untrained/missing weights | Train OCR first |
| `tests/` empty | No automated test suite | Low priority pre-competition |
| No root README | Onboarding relies on `docs/` | This handoff fixes that |
| Git mostly untracked | No commit history | Start committing soon |

---

## 17. Questions your teammate will likely ask

**Q: Where do I start coding?**  
A: `src/arm/controller.py` if hardware is available; otherwise OCR training or `main_pipeline.py` stub with demo data.

**Q: Can I use GPU?**  
A: No. Competition laptop has no discrete GPU. All torch code is CPU.

**Q: Why two `download_datasets.py` files?**  
A: Root one is a simple 40-line shortcut. `scripts/download_datasets.py` is the full CLI version. Prefer scripts/.

**Q: Why does `verify_all.py` fail?**  
A: Expected until models trained and controller.py exists. Use section-by-section checks.

**Q: What's `.cursorrules`?**  
A: Architecture contract for AI assistants and humans. Read before changing design.

---

## 18. Documents in `docs/`

| File | For whom | Purpose |
|------|----------|---------|
| `SETUP.md` | New developer | Install everything on a fresh PC |
| `TECHNICAL_HANDOFF.md` | You → teammate | This file — big picture + priorities |
| `CODEBASE_WALKTHROUGH.md` | Developer | Every file, every block, when to run |
| `../AXUM_CLAUDE_BRIEFING.md` | External AI chats | Full project context for Claude/GPT |

---

## 19. One-page summary to read aloud

> "The robot's brain is a Python project on a Windows laptop. Arduino moves the hardware over USB serial — that Python wrapper doesn't exist yet and is job one. Most of the smart stuff — reading Ge'ez, finding cracks, mapping salt, advising conservation, grouping fragments, making PDF catalogues — is already written as separate modules in `src/`. What's missing is the mission script that calls them in order and talks to the dashboard. The OCR dataset is downloaded; we need to train the model. Classifier needs more images. Demo catalogue and dashboard work today without the robot. Read SETUP, then build controller.py, then main_pipeline.py, train OCR in parallel."

---

*Last updated: June 2026. Update this file when controller.py or main_pipeline.py land.*
