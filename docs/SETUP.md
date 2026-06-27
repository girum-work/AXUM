# AXUM Rover — Developer Setup Guide

Use this document when onboarding a new developer machine. Follow every step in order.

**Target platform:** Windows 10/11 (same as the competition laptop)  
**Project root:** `C:\Users\<you>\Documents\Programming\Projects\AXUM`  
**Python:** 3.12.x (tested on 3.12.10)

---

## 0. What you need before starting

| Requirement | Why |
|-------------|-----|
| **Python 3.12** | All code and venv are 3.12 |
| **Git** | Pull the codebase |
| **~15 GB free disk** | OCR datasets alone are several GB |
| **Internet** | First-time pip + optional dataset download |
| **Arduino IDE** (optional now) | Flash Mega + ESP32 firmware later |
| **LM Studio** (optional) | Local LLM for Ge'ez text restoration |
| **Meshroom** (optional) | Real 3D photogrammetry; demo meshes work without it |

**Not required:** CUDA, GPU drivers, Raspberry Pi, cloud accounts.

---

## 1. Clone the repository

```powershell
cd C:\Users\<you>\Documents\Programming\Projects
git clone <your-repo-url> AXUM
cd AXUM
```

If you receive the project as a zip, extract it to the same path and open that folder in your editor.

---

## 2. Install Python 3.12

1. Download from [python.org](https://www.python.org/downloads/)
2. During install, check **“Add python.exe to PATH”**
3. Verify:

```powershell
python --version
# Expected: Python 3.12.x
```

---

## 3. Create and activate the virtual environment

Always work inside `venv`. Never install packages globally.

```powershell
cd C:\Users\<you>\Documents\Programming\Projects\AXUM
python -m venv venv
.\venv\Scripts\activate
```

You should see `(venv)` in your prompt. Reactivate this every time you open a new terminal:

```powershell
cd C:\Users\<you>\Documents\Programming\Projects\AXUM
.\venv\Scripts\activate
```

---

## 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## 5. Install PyTorch (CPU only)

Install PyTorch **before** the rest of `requirements.txt`. This project must run without a GPU.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify CPU mode:

```powershell
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# CUDA: False  ← must be False
```

---

## 6. Install remaining Python dependencies

```powershell
pip install -r requirements.txt
```

**Note:** `requirements.txt` has duplicate entries from iterative development. If pip reports version conflicts, prefer the **already-installed torch** from step 5 and re-run:

```powershell
pip install ultralytics datasets huggingface_hub pyarrow pandas openai
```

### Packages that matter most

| Package | Used for |
|---------|----------|
| `opencv-python` | Crack detection, imaging, camera frames |
| `torch` / `torchvision` | OCR, classifier, all ML |
| `ultralytics` | YOLOv8 artefact + coin detection |
| `flask` / `flask-socketio` | Dashboard server |
| `pyserial` | Arduino communication (when `controller.py` exists) |
| `reportlab` / `qrcode` | PDF catalogue |
| `loguru` | All logging in `src/` |
| `albumentations` | OCR training augmentations |
| `datasets` / `huggingface_hub` / `pyarrow` | HHD-Ethiopic dataset download |
| `requests` | ESP32-CAM HTTP capture |
| `librosa` / `sounddevice` | Future acoustic sensing |
| `pyttsx3` | Future Ge'ez TTS |
| `sqlalchemy` | Planned heritage database |

### Optional packages (install when needed)

```powershell
pip install openai          # LM Studio compatibility check in verify_all.py
pip install open3d          # Fragment grouper ICP (falls back without it)
```

---

## 7. Verify the install

```powershell
python verify_install.py
```

Expected output ends with:

```
ALL LIBRARIES INSTALLED SUCCESSFULLY
Ready to build.
```

If anything fails, fix that package before continuing.

---

## 8. Configure machine-specific paths

Edit `config.py` at the project root. **Every developer must update these:**

| Constant | What to change |
|----------|----------------|
| `SERIAL_PORT` | Your Arduino COM port (Device Manager → Ports) e.g. `"COM5"` |
| `ESP32_CAM_URL` | ESP32 IP after WiFi boot e.g. `"http://192.168.1.105:81/stream"` |
| `MESHROOM_PATH` | Path to Meshroom.exe if installed, or leave and use demo meshes |

Do **not** hardcode paths elsewhere — only in `config.py`.

---

## 9. Data and models (what to download vs what ships with repo)

### If the repo already includes `data/geez_characters/` (~80k images)

You can skip dataset download and go straight to training:

```powershell
python scripts/train_ocr.py
```

### If OCR data is missing

**Option A — quick root script:**

```powershell
python download_datasets.py
```

**Option B — full CLI (recommended):**

```powershell
python scripts/download_datasets.py
python scripts/extract_cached_dataset.py
```

This downloads HHD-Ethiopic from HuggingFace (~30–90 minutes). Do not close the terminal.

### Artefact classifier images

```powershell
python scripts/download_artefact_images.py
python scripts/download_artefact_images.py --report-only   # check counts
```

Target: 500+ images across 5 classes (see `config.py` → `ARTEFACT_CLASS_MIN_COUNTS`).

### Trained model weights (`models/`)

The repo may ship with an **empty** `models/` folder. You must train or copy weights:

| File | How to create |
|------|----------------|
| `models/geez_ocr.pth` | `python scripts/train_ocr.py` (hours on CPU) |
| `models/artefact_classifier.pth` | `python scripts/train_classifier.py` |
| `models/yolov8_artefacts.pt` | `python scripts/train_yolov8.py` |
| `models/yolov8_coins.pt` | `python scripts/train_yolov8_coins.py` |

### Demo data (works without hardware)

```powershell
python scripts/seed_demo_catalogue.py
python scripts/generate_demo_meshes.py
```

Creates catalogue JSON/PDF and OBJ meshes for dashboard testing.

---

## 10. Smoke-test the software stack

Run from project root with venv active:

```powershell
# 1. Library check
python verify_install.py

# 2. Module spike tests (no hardware)
python scripts/test_crack_spike.py
python scripts/test_ocr_spike.py

# 3. Dashboard (keep running, open browser)
python src/dashboard/server.py
# → http://localhost:5000

# 4. Full pre-mission check (will fail on missing controller/models until built)
python scripts/verify_all.py
```

---

## 11. Optional: LM Studio (Ge'ez text restoration)

1. Install [LM Studio](https://lmstudio.ai/) desktop app
2. Download **Qwen2.5-1.5B-Instruct** (or similar small model)
3. Start the **local server** on port `1234`
4. Test:

```powershell
python scripts/test_llm_spike.py
```

Config: `LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"` in `config.py`.

---

## 12. Optional: Meshroom (real 3D reconstruction)

1. Install AliceVision Meshroom for Windows
2. Set `MESHROOM_PATH` in `config.py`
3. Place turntable photos in `scans/photos/<object_id>/`
4. Run:

```powershell
python scripts/run_meshroom.py --object-id AXUM-OBJ-001
```

Without Meshroom, use `python scripts/generate_demo_meshes.py` for dashboard demos.

---

## 13. Optional: Arduino + ESP32 firmware

1. Install [Arduino IDE 2.x](https://www.arduino.cc/en/software)
2. Flash **`arduino/axum_rover/axum_rover.ino`** to **Arduino Mega 2560**
3. Flash **`arduino/esp32_cam/esp32_cam.ino`** to **ESP32-CAM** — set `WIFI_SSID` / `WIFI_PASSWORD` first
4. Connect Mega via USB; note COM port → update `config.py`
5. Hardware Python control requires **`src/arm/controller.py`** (not written yet — see handoff doc)

---

## 14. IDE setup (recommended)

- **Editor:** Cursor or VS Code
- **Python interpreter:** `./venv/Scripts/python.exe`
- **Read first:** `.cursorrules` (architecture contract)
- **Handoff docs:** `docs/TECHNICAL_HANDOFF.md`, `docs/CODEBASE_WALKTHROUGH.md`

---

## 15. Common problems

### `ModuleNotFoundError: No module named 'src'`

Run commands from **project root**, not from inside `src/`:

```powershell
cd C:\Users\<you>\Documents\Programming\Projects\AXUM
python scripts/train_ocr.py
```

### `ImportError: cannot import name 'ArduinoSerial' from 'src.arm.controller'`

`src/arm/controller.py` does not exist yet. This is expected until the hardware layer is implemented.

### PyTorch installs CUDA version by mistake

Uninstall and reinstall CPU build:

```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### `verify_all.py` fails on LM Studio

Optional for now. Start LM Studio server or ignore until working on restoration.

### Dataset download interrupted

Re-run `python scripts/download_datasets.py` — HuggingFace cache resumes partial downloads.

### Training is very slow

Normal on CPU. OCR training may take many hours. Use fewer epochs for first test:

```powershell
python scripts/train_ocr.py --epochs 5
```

---

## 16. Onboarding checklist

Copy this for your teammate:

- [ ] Python 3.12 installed
- [ ] Repo cloned
- [ ] `venv` created and activated
- [ ] PyTorch CPU installed (`CUDA: False`)
- [ ] `pip install -r requirements.txt` succeeded
- [ ] `python verify_install.py` passes
- [ ] Read `.cursorrules`
- [ ] Read `docs/TECHNICAL_HANDOFF.md`
- [ ] Updated `config.py` (COM port, ESP32 URL)
- [ ] Demo dashboard runs: `python src/dashboard/server.py`
- [ ] Understands P0 tasks: `controller.py`, `main_pipeline.py`, OCR training

---

## 17. Daily workflow

```powershell
cd C:\Users\<you>\Documents\Programming\Projects\AXUM
.\venv\Scripts\activate

# Edit code in src/ or scripts/
# Run the relevant script or module __main__ test
python src/imaging/salt_mapper.py          # example: synthetic self-test

# Before demo day
python scripts/verify_all.py
```

**Rules:** Never commit `venv/`. Never use GPU code. Never hardcode paths outside `config.py`.
