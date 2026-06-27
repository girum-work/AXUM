# AXUM Rover — Codebase Walkthrough

Block-by-block guide to **every project file**. Read alongside the source.

**Legend:**
- **EXEC** = run directly (`python path/to/file.py`)
- **IMPORT** = library module; imported by other code
- **CLI** = command-line script in `scripts/`
- **CONFIG** = edit values, never execute
- **FIRMWARE** = flash with Arduino IDE
- **MISSING** = referenced but not on disk yet

---

## Table of contents

1. [Root files](#1-root-files)
2. [config.py — full walkthrough](#2-configpy--full-walkthrough)
3. [src/arm](#3-srcarm)
4. [src/analysis](#4-srcanalysis)
5. [src/catalogue](#5-srccatalogue)
6. [src/crack_detection](#6-srccrack_detection)
7. [src/dashboard](#7-srcdashboard)
8. [src/imaging](#8-srcimaging)
9. [src/object_detection](#9-srcobject_detection)
10. [src/ocr](#10-srcocr)
11. [src/photogrammetry](#11-srcphotogrammetry)
12. [src/pipeline](#12-srcpipeline)
13. [scripts/ — all scripts](#13-scripts--all-scripts)
14. [arduino/](#14-arduino)
15. [data/ directories](#15-data-directories)
16. [Missing files (planned)](#16-missing-files-planned)

---

## 1. Root files

### `config.py` — CONFIG
See [§2 dedicated walkthrough](#2-configpy--full-walkthrough).

### `requirements.txt` — CONFIG
Pinned Python packages. Install with `pip install -r requirements.txt` **after** CPU torch (see `SETUP.md`). Contains duplicate blocks from development — pip resolves to last matching entry.

### `verify_install.py` — EXEC (once after install)
| Block | What it does |
|-------|----------------|
| Imports | Tries importing cv2, numpy, torch, torchvision, matplotlib, sklearn, librosa, serial, flask, reportlab, sqlalchemy |
| CUDA check | Prints `torch.cuda.is_available()` — must be `False` |
| Summary | Prints ALL LIBRARIES INSTALLED or lists failures |

**When:** After `pip install`, before any development.

### `download_datasets.py` — EXEC (once if OCR data missing)
| Block | What it does |
|-------|----------------|
| `sys.path.insert` | Adds project root so `from src.ocr.pipeline import ...` works |
| `download_hhd_ethiopic()` | Downloads HHD-Ethiopic from HuggingFace to `data/geez_characters/` |
| `verify_dataset()` | Counts images; expects >500 |

**When:** Fresh clone without `data/geez_characters/`. Prefer `scripts/download_datasets.py` for CLI options.

### `.cursorrules` — READ ONLY
Architecture contract: hardware pins, serial protocol, coding rules, things to never suggest. **Read before writing any code.**

### `AXUM_CLAUDE_BRIEFING.md` — READ ONLY
Full project context for pasting into AI chat sessions.

---

## 2. `config.py` — full walkthrough

**Purpose:** Single source of truth. Every module imports constants from here.

### Lines 1–12 — Paths
```python
ROOT_DIR = Path(__file__).parent   # Project root (folder containing config.py)
DATA_DIR = ROOT_DIR / "data"       # All datasets, KB, catalogue JSON
MODELS_DIR = ROOT_DIR / "models"   # Trained .pth / .pt weights
SCANS_DIR = ROOT_DIR / "scans"     # Runtime scan outputs
SCAN_PHOTOS_DIR = SCANS_DIR / "photos"   # Turntable JPEGs per object_id
MESH_DIR = SCANS_DIR / "meshes"         # Published OBJ for dashboard
CATALOGUE_DIR = DATA_DIR / "catalogue"   # Per-object JSON + PDF
```
**Why:** Centralizes paths so code works on any machine after clone.

### Lines 15–19 — Camera
```python
ESP32_CAM_URL = "http://192.168.1.105:81/stream"  # CHANGE to your ESP32 IP
CAMERA_WIDTH/HEIGHT/FPS  # Expected stream resolution
```
**When to edit:** After flashing ESP32-CAM and reading IP from Serial Monitor.

### Lines 21–29 — Crack detection
OpenCV parameters: Canny thresholds, minimum crack area/length, aspect ratio filter, severity threshold 0.4 (below = treatable).

**Used by:** `src/crack_detection/detector.py`

### Lines 31–35 — Heatmap wall
Grid size for future `resurrection_wall.py` visualization (10×15 cells, 80cm wall).

### Lines 37–48 — OCR (active values)
**Note:** Lines 40–41 are **obsolete** (64×64, 231 classes). Lines 42–48 **override** them:
```python
OCR_IMG_SIZE = (32, 128)      # height × width — character strip shape
NUM_GEEZ_CLASSES = 300        # charset size after build_geez_charset()
OCR_BEAM_WIDTH = 5            # CTC beam search at inference
OCR_USE_WEIGHTED_SAMPLER = True   # training: oversample rare Ge'ez chars
OCR_USE_STONE_AUGMENT = True      # training: stone-specific albumentations
OCR_USE_ADAPTIVE_BINARIZE = True  # inference: CLAHE + adaptive threshold
```
**Used by:** `src/ocr/model.py`, `src/ocr/pipeline.py`, `scripts/train_ocr.py`

### Lines 50–54 — Model paths
```python
OCR_MODEL_PATH = MODELS_DIR / "geez_ocr.pth"
OBJ_MODEL_PATH = MODELS_DIR / "artefact_classifier.pth"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8_artefacts.pt"
```

### Lines 56–91 — Object detection + artefact dataset
- `OBJ_CLASSES` — 5 classifier labels
- `ARTEFACT_CLASS_MIN_COUNTS` / `IDEAL_COUNTS` — download targets for `download_artefact_images.py`
- Quality filters: min 224px, 40% coverage, sharpness ≥60, max 3 images per museum object
- Museum API endpoints (Met, British Museum, Smithsonian)

### Lines 93–116 — Coin YOLO second stage
Seven coin era classes covering all Ethiopian numismatics. Heuristics for worn coins → ancient + OCR.

### Lines 118–125 — Arduino + turntable
```python
SERIAL_PORT = "COM3"     # CHANGE per machine
TURNTABLE_STEPS = 36     # photos per 360° (10° each)
TURNTABLE_SETTLE_MS = 500  # pause before PHOTO after step
```

### Lines 127–141 — Meshroom photogrammetry
Meshroom exe path, cache dir, OBJ filename priority list, 2-hour timeout.

### Lines 143–162 — LLM restoration training/inference
Paths to JSONL corpora, damage injection rates, LM Studio URL `http://127.0.0.1:1234/v1`.

### Lines 164–186 — Databases, treatment, fragments
Conservation KB path, substrate mapping from classifier label → stone type, fragment match thresholds (65% possible, 85% confirmed).

### Lines 188–203 — LED, dashboard, logging
NeoPixel config, Flask host/port 5000, log file auto-creates `logs/axum_rover.log`.

---

## 3. `src/arm`

### `__init__.py` — IMPORT
Re-exports `ArduinoSerial`, `ArmController`, `TurntableController`, `MeshroomInterface` from **`controller.py` (MISSING)**.

**Impact:** Any `from src.arm import ArduinoSerial` crashes until controller is written.

### `controller.py` — MISSING — must implement

Expected structure (from `.cursorrules` + `verify_all.py`):

| Class | Responsibility |
|-------|----------------|
| `ArduinoSerial` | `send_command(cmd) → response`, `_find_arduino_port()`, retry logic |
| `ArmController` | `go_pose(name)`, `set_grip(angle)`, wraps ARM/GRIP/POSE commands |
| `TurntableController` | `step(n)`, `rotate_degrees(deg)`, coordinates PHOTO + settle delay |
| `CameraInterface` | HTTP to ESP32 `/capture` and stream URL from config |

**Serial rule:** Never use raw `serial.write()` outside this file.

---

## 4. `src/analysis`

### `__init__.py` — IMPORT
Lazy-imports treatment advisor and fragment grouper symbols via `__getattr__` (avoids circular imports).

### `treatment_advisor.py` — IMPORT + EXEC (`__main__`)
| Section | Contents |
|---------|----------|
| §1 Data structures | `DiagnosticInputs`, `TreatmentProtocol`, `SafeTreatment`, `DangerousTreatment` |
| §2 KB loader | Reads `conservation_kb.json`, indexes substrates and decay patterns |
| §3 Scoring | Maps sensor inputs → decay pattern probabilities |
| §4 Protocol builder | Ranks safe treatments, flags dangerous ones with harm mechanisms |
| §5 `run_treatment_advisor()` | Main entry — pass `DiagnosticInputs`, get `TreatmentProtocol` |
| §6 `__main__` | Synthetic pottery + high salt scenario — prints protocol |

**When to run standalone:** Testing conservation logic without robot.

**Called by:** Future `main_pipeline.py`, dashboard via `emit_treatment_protocol()`.

### `fragment_grouper.py` — IMPORT + EXEC
| Section | Contents |
|---------|----------|
| §1 | `FragmentGroup` dataclass |
| §2 | Material features: colour histogram, UV signature, density |
| §3 | `compute_match_score()` — weighted blend of geometry, material, class, inscription |
| §4 | Open3D ICP optional for fracture surface alignment |
| §5 | `FragmentGrouper` — persistent store in `fragment_groups.json` |
| §6 | `register_object()` — called by `CatalogueService` after each scan |

**Why:** Three pottery shards should appear as one vessel in the catalogue.

---

## 5. `src/catalogue`

### `__init__.py` — IMPORT
Exports `CatalogueGenerator`, `ObjectRecord`, `CatalogueService`.

### `records.py` — IMPORT
**`ObjectRecord`** dataclass — the schema for one scanned artefact. Fields filled incrementally by pipeline stages. Methods: `to_dict()`, `from_dict()`.

**Critical fields:** `object_id`, `class_name`, `inscription_text`, `crack_severity`, `mesh_path`, `group_id`, `interventions`.

### `generator.py` — IMPORT
**`CatalogueGenerator`** class:
- `add_entry(record)` — appends one A4 PDF page + saves `{object_id}.json`
- `_regenerate_pdf()` — rebuilds full `axum_catalogue.pdf` with dark museum styling (gold/navy palette)
- Uses ReportLab — no external LaTeX

**Why PDF:** Judges see professional museum output during demo.

### `mesh_registry.py` — IMPORT
- `enrich_catalogue_entry()` — adds `mesh_ready`, `mesh_url` for dashboard 3D viewer
- `apply_publish_result()` — writes mesh path after photogrammetry

### `service.py` — IMPORT
**`CatalogueService`** — high-level API:
```python
service = CatalogueService()
service.register_object(record)  # → fragment grouping + JSON + PDF
service.load_all_objects()       # → dashboard catalogue tab
service.update_mesh(object_id, mesh_path)
```

**Orchestration point:** Pipeline calls this once per artefact when processing completes.

---

## 6. `src/crack_detection`

### `__init__.py` — IMPORT
Pointer comment: run `detector.py` for standalone test.

### `detector.py` — IMPORT + EXEC (~829 lines)
| Section | Contents |
|---------|----------|
| §1 Preprocessing | Grayscale, Gaussian blur, optional auto-Canny |
| §2 Crack extraction | Morphological thinning, contour filter by aspect ratio + area |
| §3 Severity scoring | Normalized 0–1 from depth proxy + crack density |
| §4 Heatmap overlay | Draws cracks on image for dashboard |
| §5 `run_crack_detection()` | Main API — image in, `CrackResult` out |
| §6 `__main__` | Synthetic cracked plate image — saves overlay PNG |

**No ML.** Pure OpenCV. Parameters from `config.py` (`CANNY_T1`, `MIN_CRACK_AREA`, etc.).

**When to EXEC:** Verify OpenCV pipeline before real artefact photos.

---

## 7. `src/dashboard`

### `server.py` — EXEC
| Section | Contents |
|---------|----------|
| Flask app setup | Templates in `templates/`, port from config |
| `mission_state` dict | Global state: status, log, catalogue list |
| `emit_event()` | **The only function pipeline needs** — pushes WebSocket events |
| Helper emitters | `emit_camera_frame()`, `emit_inscription()`, `emit_treatment_protocol()`, etc. |
| HTTP routes | `/`, `/catalogue`, `/api/objects`, `/models/<id>/model.obj` |
| SocketIO handlers | Client connect, request catalogue refresh |

**Run:** `python src/dashboard/server.py` → `http://localhost:5000`

**Event names pipeline should emit:**
`mission_started`, `artefact_picked`, `scan_started`, `crack_detected`, `salt_detected`, `inscription_recognized`, `translation_ready`, `treatment_protocol`, `artefact_complete`, `mission_complete`, `error`

### `templates/dashboard.html` — UI
Live mission view: camera feed, analysis panels, log. Connects via Socket.IO.

### `templates/catalogue.html` — UI
Browse scanned objects, 3D mesh viewer (loads OBJ from `/models/` route).

---

## 8. `src/imaging`

### `salt_mapper.py` — IMPORT + EXEC (~1145 lines)
| Section | Contents |
|---------|----------|
| §1 Data structures | `SaltZone`, `SaltMapResult` |
| §2 Hardware capture | UV LED trigger (future Arduino), ESP32 UV image, conductivity probe |
| §3 UV fluorescence | Detects bright regions under 365nm — salt glows |
| §4 Conductivity | Confirms salt inside stone vs surface dust |
| §5 Risk levels 0–4 | None → critical migration |
| §6 Grid mapping | Maps zones to artefact surface grid |
| §7 `run_salt_mapping()` | Main entry |
| §8 `__main__` | Synthetic UV image — no hardware |

**Hardware note:** Conductivity via STM32 ADC over I2C — not wired in Python yet.

### Missing (planned)
- `photometric_stereo.py` — 4-quadrant NeoPixel depth
- `multispectral.py` — NDCI stress index from UV/IR

---

## 9. `src/object_detection`

### `detector.py` — IMPORT + EXEC (~996 lines)
| Section | Contents |
|---------|----------|
| §1 Preprocessing | ImageNet normalize, 224×224 |
| §2 `ArtefactClassifier` | MobileNetV2 + custom head, 5 classes |
| §3 Training functions | `train_artefact_classifier()`, `ArtefactDataset` |
| §4 `YOLOArtefactClassifier` | Ultralytics wrapper, COCO fallback |
| §5 Stage 1 detection | `detect_objects_in_tray()` — OpenCV contours on tray background |
| §6 Stage 2 classify | Crop each contour → classifier or YOLO |
| §7 `detect_and_classify_tray()` | Full pipeline — list of detections with class + confidence |
| §8 Visualization | `draw_detection_overlay()` |

**Why two stages:** Contours give precise arm coordinates; classifier gives category.

### `coin_inspection.py` — IMPORT
Second stage when class = `coin`:
- YOLO coin subtype (7 era classes)
- Wear heuristic → `coin_wear_unknown_ancient`
- Recommends Ge'ez OCR when inscription crop confidence low

### `artefact_label_rules.py` — IMPORT
Post-classifier rules: e.g. override labels based on size/shape heuristics, competition-specific label normalization.

---

## 10. `src/ocr`

### `model.py` — IMPORT (~598 lines)
| Section | Contents |
|---------|----------|
| §1 Charset | `build_geez_charset()` — Unicode U+1200–U+137F |
| §2 CNN backbone | Stacked conv blocks, adaptive pool |
| §3 BiLSTM | Bidirectional sequence modeling |
| §4 CTC head | Linear → per-frame class logits |
| §5 `GeezOCRModel` | Full nn.Module |
| §6 Decode | Greedy + beam search (`OCR_BEAM_WIDTH`) |
| §7 Save/load | `save_ocr_model()`, `load_ocr_model()` |

### `pipeline.py` — IMPORT (~1570 lines)
| Section | Contents |
|---------|----------|
| §1 Preprocess | `preprocess_for_ocr()` — CLAHE, adaptive binarize |
| §2 Dataset | `HHDEthiopicDataset` — reads CSV + image paths |
| §3 Augmentations | Stone-specific albumentations pipeline |
| §4 Training | `train_ocr_model()` — CTC loss, weighted sampler |
| §5 Inference | `GeezOCRPipeline.predict()` — text + confidence |
| §6 Text regions | Contour-based inscription region finder |
| §7 Download | `download_hhd_ethiopic()`, `verify_dataset()` |

**Training entry:** `scripts/train_ocr.py` calls `train_ocr_model()`.

### `llm_restoration.py` — IMPORT (~963 lines)
| Section | Contents |
|---------|----------|
| §1 `RestorationResult` | restored text, translation, confidence, mode |
| §2 Known phrases | Regex database of common inscription formulas |
| §3 LM Studio client | OpenAI-compatible API to local Qwen |
| §4 Ollama few-shot | Alternative local LLM path |
| §5 Rule-based fallback | No LLM — pattern completion only |
| §6 `restore_inscription()` | Main API |

### Supporting OCR files
| File | Role |
|------|------|
| `corpus.py` | Build JSONL training data for LLM fine-tune from HHD phrases |
| `damage.py` | Inject synthetic `[MISSING]` tokens and erosion for training |
| `ocr_postprocess.py` | Clean OCR output, merge broken characters |
| `restoration_prompts.py` | System/user prompt templates + few-shot examples |

---

## 11. `src/photogrammetry`

### `meshroom.py` — IMPORT + EXEC (~658 lines)
| Section | Contents |
|---------|----------|
| §1 `MeshPublishResult` | Paths, vertex/face counts, warnings |
| §2 Path helpers | `object_photo_dir()`, `object_mesh_dir()` |
| §3 Meshroom subprocess | Runs `Meshroom.exe` batch on photo folder |
| §4 Export picker | Chooses best OBJ from export tree (`texturedMesh.obj` priority) |
| §5 Publish | Copies OBJ+MTL+textures to `scans/meshes/<id>/model.obj` |
| §6 `generate_demo_mesh()` | Procedural smooth OBJ when Meshroom unavailable |
| §7 `run_photogrammetry()` | Main API |

### `mesh_stage.py` — IMPORT
Thin wrapper connecting photogrammetry → catalogue:
```python
process_object_mesh("AXUM-OBJ-001")  # → MeshPublishResult + updated ObjectRecord
process_demo_meshes()                 # → demo OBJ for all catalogue entries
```

---

## 12. `src/pipeline`

### `__init__.py` — IMPORT
Lazy `__getattr__`:
- `process_object_mesh`, `process_demo_meshes` → from `mesh_stage.py` ✅
- `MissionPipeline`, `MissionState`, etc. → from `main_pipeline.py` ❌ MISSING

### `main_pipeline.py` — MISSING
Must implement mission loop described in `TECHNICAL_HANDOFF.md` §5.

---

## 13. `scripts/ — all scripts`

### Dataset download & preparation

| Script | EXEC when | What it does |
|--------|-----------|--------------|
| `download_datasets.py` | OCR data missing | Full HF download + verify (CLI args) |
| `download_geez_dataset.py` | Alternative Ge'ez source | Short wrapper |
| `extract_cached_dataset.py` | After HF snapshot | Converts cache → folder structure |
| `merge_datasets.py` | Combining OCR sources | Merges geez_chars into geez_merged |
| `download_artefact_images.py` | Before classifier train | Met/Wikimedia/SI APIs, quality filters |
| `cleanup_artefact_dataset.py` | After bad downloads | Removes duplicates, moves rejects to `_rejected/` |

### Training

| Script | EXEC when | Output | Runtime |
|--------|-----------|--------|---------|
| `train_ocr.py` | OCR data ready | `models/geez_ocr.pth` | Hours (CPU) |
| `train_classifier.py` | 500+ artefact imgs | `models/artefact_classifier.pth` | 1–3 h |
| `train_yolov8.py` | Detection labels ready | `models/yolov8_artefacts.pt` | 1–2 h |
| `train_yolov8_coins.py` | Coin images ready | `models/yolov8_coins.pt` | 1–2 h |

### LLM / restoration data

| Script | EXEC when | Output |
|--------|-----------|--------|
| `generate_dataset.py` | Building restoration corpus | JSONL with synthetic damage |
| `export_colab_training_data.py` | Before Colab fine-tune | Zip for upload |
| `export_restoration_colab.py` | Colab QLoRA prep | Training files |

### Demo & photogrammetry

| Script | EXEC when | Output |
|--------|-----------|--------|
| `seed_demo_catalogue.py` | Fresh clone / dashboard test | 4× JSON + PDF in `data/catalogue/` |
| `generate_demo_meshes.py` | Dashboard 3D test | OBJ in `scans/meshes/` |
| `run_meshroom.py` | Real turntable photos exist | Published mesh |
| `register_mesh.py` | Manual mesh import | Updates catalogue mesh_path |

### Verification & spikes

| Script | EXEC when | Purpose |
|--------|-----------|---------|
| `verify_all.py` | Pre-demo | Models, data, LM Studio, Arduino |
| `test_crack_spike.py` | After install | Import crack detector |
| `test_ocr_spike.py` | After install | Import OCR stack |
| `test_llm_spike.py` | LM Studio running | Test restoration API |
| `test_ocr.py` | After OCR train | Sample inference |
| `test_ocr_samples.py` | OCR debugging | Batch sample images |
| `run_ocr_ablation.py` | OCR tuning | Compare preprocess flags |
| `analyze_ocr_errors.py` | After OCR eval | Error breakdown by character |

### Internal / one-off

| Script | Note |
|--------|------|
| `build_conservation_kb.py` | Already run — generated `conservation_kb.json` |
| `_write_kb_fragments.py` | KB section builder — dev tool |
| `_probe_*.py` | Museum API exploration — not for production |

---

## 14. `arduino/`

### `axum_rover/axum_rover.ino` — FIRMWARE
| Block | Contents |
|-------|----------|
| Pin definitions | Motors, servos, stepper, NeoPixel, ultrasonics, encoders, CAM trigger |
| Servo objects | 5 servos (S0–S4) |
| Pre-computed poses | PARK, HOVER_TRAY, PICK, PLACE, SCAN arrays |
| `setup()` | Pin modes, serial begin 115200, attach servos |
| `loop()` | Read serial line → parse command → execute → respond |
| Motor functions | `drive(left, right)` with encoder interrupt ISRs |
| Arm functions | Interpolated servo moves, gripper slow close |
| Stepper | Manual half-step turntable rotation |
| NeoPixel | Full ring + quadrant modes for photometric stereo |

**Flash to:** Arduino Mega 2560  
**Calibrate:** Pose angle arrays on physical arm before competition

### `esp32_cam/esp32_cam.ino` — FIRMWARE
| Block | Contents |
|-------|----------|
| WiFi credentials | **CHANGE** `WIFI_SSID`, `WIFI_PASSWORD` |
| Camera init | AI Thinker pin map |
| HTTP servers | `/stream` (MJPEG), `/capture` (single JPEG), `/status` |
| Trigger pin | GPIO 13 — HIGH pulse from Mega pin 30 |

**Flash to:** ESP32-CAM  
**After boot:** Copy IP to `config.py` → `ESP32_CAM_URL`

---

## 15. `data/` directories

| Directory | Contents | Needed for |
|-----------|----------|------------|
| `geez_characters/` | HHD-Ethiopic ~79k PNG + CSVs | OCR training |
| `geez_chars_clean/` | 32×32 balanced chars | Optional OCR augment |
| `geez_merged/` | Merged training pairs | Alternative OCR train path |
| `hhd_raw/` | HF cache remnants | Can ignore if geez_characters complete |
| `artefact_classes/` | `{class}/*.jpg` + metadata.csv | Classifier training |
| `artefact_classes/_rejected/` | Failed quality filter images | Ignore for training |
| `corpus/geez_inscriptions.jsonl` | LLM corpus source | Restoration fine-tune |
| `geez_restoration_*.jsonl` | Train/val/eval splits | LLM fine-tune |
| `catalogue/` | AXUM-OBJ-*.json + PDF | Dashboard + demo |
| `databases/conservation_kb.json` | Treatment advisor KB | Analysis |
| `databases/fragment_groups.json` | Fragment group persistence | Analysis |
| `crack_images/` | Test images | Crack detector dev |

**Not on disk yet:** `coin_subtypes/`, `yolo_coin_dataset/`, `databases/inscriptions.json`, `databases/artefacts.json`, `databases/axum_heritage.db`

---

## 16. Missing files (planned)

| Path | Purpose | Priority |
|------|---------|----------|
| `src/arm/controller.py` | Serial hardware layer | **P0** |
| `src/pipeline/main_pipeline.py` | Mission orchestrator | **P0** |
| `src/imaging/photometric_stereo.py` | 4-light depth | P2 |
| `src/imaging/multispectral.py` | NDCI stress | P2 |
| `src/analysis/fragility_clock.py` | Years-remaining estimate | P2 |
| `src/sensing/acoustic_tap.py` | Solenoid + piezo | P3 |
| `src/sensing/void_detector.py` | Internal void detection | P3 |
| `src/intervention/*.py` | Pump, brush, reunification | P3 |
| `src/audio/geez_speaker.py` | TTS | P3 |
| `src/visualization/resurrection_wall.py` | Heatmap wall | P3 |

---

## Quick reference: which file answers which question?

| Question | Go to |
|----------|-------|
| Where is COM port configured? | `config.py` → `SERIAL_PORT` |
| How to send STOP to robot? | `controller.py` (to write) → `send_command("STOP")` |
| How to train OCR? | `scripts/train_ocr.py` |
| How to add catalogue entry? | `catalogue/service.py` → `register_object()` |
| How to push dashboard update? | `dashboard/server.py` → `emit_event()` |
| What's the artefact data schema? | `catalogue/records.py` → `ObjectRecord` |
| How to run 3D mesh pipeline? | `pipeline/mesh_stage.py` |
| Conservation safe/unsafe rules? | `analysis/treatment_advisor.py` + `conservation_kb.json` |
| Serial command list? | `.cursorrules` or `arduino/axum_rover.ino` header |

---

*For install steps see `SETUP.md`. For priorities see `TECHNICAL_HANDOFF.md`.*
