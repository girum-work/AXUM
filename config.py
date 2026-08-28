# config.py
"""
AXUM ROVER — Central Configuration
===================================
WHAT:  Every path, threshold, hardware constant, and model filename for the project.
WHY:   Single source of truth — no magic numbers in src/ modules.
WHO:   Edit this file when moving machines (COM port, ESP32 IP, Meshroom path).
HOW:   `from config import SCAN_PHOTOS_DIR, OCR_CONFIDENCE_MIN` everywhere else.

Never hardcode paths or thresholds outside this file.
See docs/CODEBASE_WALKTHROUGH.md §2 for a line-by-line guide.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────
# All runtime directories are relative to ROOT_DIR (this file's folder).
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models"
SCANS_DIR       = ROOT_DIR / "scans"
SCAN_PHOTOS_DIR = SCANS_DIR / "photos"       # turntable capture output
MESH_DIR        = SCANS_DIR / "meshes"       # Meshroom .obj exports
CATALOGUE_DIR   = DATA_DIR / "catalogue"     # per-object JSON + PDF
FRAGMENT_GROUPS_JSON = DATA_DIR / "databases" / "fragment_groups.json"


# ─── Camera ───────────────────────────────────────────────────
ESP32_CAM_URL   = "http://192.168.1.105:81/stream"   # update after first boot
# Pi 4 IR-CUT camera server (see services/pi/pi_camera_server.py). Not wired
# as CameraInterface's default yet -- whether the Pi camera is meant to
# fully replace ESP32_CAM_URL as primary, or run alongside it, is still an
# open integration question (see integration report). Update the IP after
# first boot, same as ESP32_CAM_URL above.
PI_CAM_URL      = "http://<pi-ip>:5001/stream"
PI_CAM_CAPTURE  = "http://<pi-ip>:5001/capture"
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FPS      = 10

# ─── Crack Detection ──────────────────────────────────────────
# Every image is resized to this long edge before detection. Filter sizes are
# in pixels, so without it a 12MP photo and a 0.3MP one are processed at
# different physical scales and the same surface scores differently.
CRACK_WORKING_EDGE = 512
# Sigmas of the ridge filter bank, in working-resolution pixels. A ridge filter
# sees structures up to roughly 2*sigma wide, so this bank covers 2-32px at
# 512px. The bank must span the widths that matter: at a top sigma of 8 the deep
# joints between rock slabs were missed entirely, for the same reason the old
# 15x15 black-hat missed them.
CRACK_RIDGE_SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)
# ABSOLUTE floor on the ridge response, not a percentile. A percentile keeps a
# fixed share of pixels on every image, so an undamaged surface still returns a
# full mask; a floor lets a clean rock return nothing. A uniform plate responds
# 0.0000, so it stays empty.
CRACK_RIDGE_THRESHOLD = 0.005
# ...but a floor alone does not transfer between surfaces. Median ridge response
# measured 0.0011 on MCS marble and 0.0252-0.0321 on natural stone and concrete,
# so 0.005 marked 95% of a concrete wall. The seed threshold is therefore the
# larger of the floor and this multiple of the image's own median response: a
# crack must stand out from THIS surface's texture as well as clear an absolute
# minimum. Unlike a percentile the retained fraction is free to be zero.
# Measured: multiple 3 costs MCS F1 nothing (0.346 -> 0.347) and cuts the wall
# photo's seeds from 95.2% to 2.8% of pixels.
CRACK_TEXTURE_MULTIPLE = 3.0
# Flattened intensity below which a pixel may be grown into a crack. 128 is
# exactly the local background, so 118 means "at least 8% darker than its
# surroundings". Used only to extend a region that a ridge seed already found.
CRACK_DARKNESS_CUT = 118.0
# Shape gate, in working-resolution pixels. Lichen, grain and sensor noise are
# dark but compact; a crack is long and thin. These reject the former.
CRACK_MIN_EXTENT_PX = 16
# MCS crack bodies measure 21.9px wide, so the first value tried here (20)
# rejected real cracks: MCS F1 0.223 at 20, 0.347 at 40, 0.435 at 80. 80 also
# lets a shadow band through on photographs (one test image jumped to 28.8% of
# pixels), so 40 is the point where width stops costing recall and starts
# costing precision.
CRACK_MAX_MEAN_WIDTH_PX = 40
MIN_CRACK_AREA  = 30
MIN_CRACK_LEN   = 40
MIN_ASPECT_RATIO= 3.0
CRACK_SEVERITY_THRESHOLD = 0.4  # below = treatable, above = flag
# Retained for the legacy Canny path, kept reachable via CrackDetector(method=)
BLUR_KERNEL     = (5, 5)
USE_AUTO_CANNY  = True
CANNY_T1        = 50
CANNY_T2        = 150

# ─── Heatmap ──────────────────────────────────────────────────
GRID_ROWS       = 10
GRID_COLS       = 15
WALL_WIDTH_CM   = 80.0
ROVER_SPEED_CMS = 10.0

# ─── OCR ──────────────────────────────────────────────────────
OCR_MODEL_PATH      = MODELS_DIR / "geez_ocr.pth"
OCR_CONFIDENCE_MIN  = 0.50    # below this = flag for expert review
OCR_IMG_SIZE      = (32, 256)   # height, width; aspect ratio is preserved with padding
NUM_GEEZ_CLASSES  = 360         # assigned core Ethiopic + CTC blank/unknown
OCR_CTC_SEQ_LEN   = 58          # CNN output timesteps for 32x256 input
OCR_BEAM_WIDTH    = 5           # CTC beam search width at inference
OCR_USE_BEAM_DECODE = True      # beam search vs greedy (no retrain needed)
OCR_USE_WEIGHTED_SAMPLER = False # enable only as a measured ablation
OCR_USE_STONE_AUGMENT = True    # albumentations stone-inscription train augment
OCR_USE_ADAPTIVE_BINARIZE = True  # CLAHE + adaptive threshold in preprocess_for_ocr

# ── Models ─────────────────────────────────────────────────────
YOLO_MODEL_PATH = MODELS_DIR / "yolo11_artefacts.pt"
                 # Set to None to use COCO fallback before fine-tuning
# Dedicated obstacle detector for arm-motion safety.  This must remain
# independent of artefact classification weights and training.
NAV_YOLO_WEIGHTS = "yolo11s.pt"
NAV_YOLO_CONFIDENCE_MIN = 0.35

# ─── Object Detection ─────────────────────────────────────────
OBJ_MODEL_PATH      = MODELS_DIR / "artefact_classifier.pth"
OBJ_CLASSES         = ["pottery", "stone_carving", "coin",
                        "inscription_fragment", "other"]
OBJ_CONFIDENCE_MIN  = 0.60
ARTEFACT_CLASSES_DIR = DATA_DIR / "artefact_classes"
ARTEFACT_METADATA_CSV = ARTEFACT_CLASSES_DIR / "metadata.csv"
ARTEFACT_MIN_IMAGES_PER_CLASS = 30   # train_classifier.py legacy floor
# Per-class dataset targets (turntable top-down classifier; Section 6 spec)
ARTEFACT_CLASS_MIN_COUNTS = {
    "pottery": 100,
    "stone_carving": 100,
    "coin": 100,
    "inscription_fragment": 80,
    "other": 80,
}
ARTEFACT_CLASS_IDEAL_COUNTS = {
    "pottery": 150,
    "stone_carving": 150,
    "coin": 150,
    "inscription_fragment": 120,
    "other": 120,
}
ARTEFACT_DATASET_MIN_TOTAL = 500
ARTEFACT_DATASET_IDEAL_TOTAL = 700
ARTEFACT_MAX_CLASS_RATIO = 2.0   # largest class ≤ 2× smallest
ARTEFACT_MIN_IMAGE_PX = 224
ARTEFACT_MIN_COVERAGE_RATIO = 0.40
ARTEFACT_MIN_SHARPNESS_VAR = 60.0
ARTEFACT_MAX_NEAR_DUPES_PER_SOURCE = 3
ARTEFACT_NEAR_DUPE_HAMMING = 6
# Museum open-access APIs (artefact image downloader)
SMITHSONIAN_API_KEY = "DEMO_KEY"
BM_SEARCH_API = "https://collection.britishmuseum.org/id/object"
BM_SPARQL_API = "https://collection.britishmuseum.org/sparql.json"
BM_REQUEST_TIMEOUT = 12

# ─── Coin detail YOLO (second stage after artefact class = coin) ─
# AXUM covers all Ethiopian numismatic history — not Aksum-only.
COIN_SUBTYPES_DIR = DATA_DIR / "coin_subtypes"
YOLO_COIN_DATASET_DIR = DATA_DIR / "yolo_coin_dataset"
YOLO_COIN_MODEL_PATH = MODELS_DIR / "yolo11_coins.pt"
# YOLO class names for era + denomination (train_yolo11_coins.py)
YOLO_COIN_CLASSES = [
    "coin_aksumite",
    "coin_menelik",
    "coin_haile_selassie",
    "coin_modern_birr",
    "coin_modern_cent",
    "coin_modern_other",
    "coin_wear_unknown_ancient",  # heavily worn → treat as ancient candidate
]
COIN_SUBCLASS_MIN_IMAGES = 40
# Wikimedia search queries per YOLO coin subtype (see download_artefact_images.py)
COIN_SUBCLASS_TARGETS = {c: COIN_SUBCLASS_MIN_IMAGES for c in [
    "coin_aksumite", "coin_menelik", "coin_haile_selassie",
    "coin_modern_birr", "coin_modern_cent", "coin_modern_other",
]}
# Heuristic: worn/blurred coins bias toward ancient + OCR inscription scan
COIN_DAMAGE_ANCIENT_THRESHOLD = 0.55   # 0–1 composite wear score
COIN_OCR_SCAN_CONFIDENCE_MAX = 0.45  # below this on crop → recommend Ge'ez OCR

# ─── Arm / Arduino ────────────────────────────────────────────
SERIAL_PORT     = "COM3"      # update to your Arduino port
SERIAL_BAUD     = 115200
ARM_MOVE_DELAY  = 0.02        # seconds between servo interpolation steps

# ─── Turntable ────────────────────────────────────────────────
TURNTABLE_STEPS      = 36     # photos per full rotation
TURNTABLE_SETTLE_MS  = 500    # wait after each step before photo
# Set True only after the quadrant/UV LED hardware has been bench-tested with
# the active firmware.  Scan code must not claim advanced lighting data until
# that physical integration is confirmed.
FIRMWARE_LED_READY   = False

# ─── Photogrammetry ───────────────────────────────────────────
MESHROOM_PATH        = r"C:\Users\Len\Downloads\Meshroom-2025.1.0-Windows\Meshroom-2025.1.0\meshroom_batch.exe"
MESHROOM_CACHE_DIR   = SCANS_DIR / "meshroom_cache"  # per-object Meshroom output
MESH_PUBLISH_NAME    = "model.obj"                   # canonical dashboard OBJ name
MESH_PUBLISH_MTL     = "model.mtl"                   # paired MTL (if textured)
# Prefer these filenames when scanning a Meshroom export tree
MESHROOM_OBJ_PRIORITY = (
    "texturedMesh.obj",
    "textured.obj",
    "model.obj",
    "mesh.obj",
)
MESHROOM_TIMEOUT_SEC = 7200                          # 2 h max per reconstruction
MESH_SHOW_PLACEHOLDER = True                         # dashboard fallback when no mesh
MESH_TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")
# Renderable model files, in preference order. OBJ first so a Meshroom
# export keeps winning over a GLB sitting in the same folder -- the OBJ
# carries the MTL and texture files the viewer already resolves.
MESH_MODEL_EXTENSIONS = (".obj", ".glb")
# Reconstruction featured on the control dashboard's preview panel. Any
# folder under MESH_DIR works; TEST-SCEAUX is the known-good textured
# result from the openMVG Sceaux Castle benchmark set.
DASHBOARD_FEATURE_MESH = "TEST-SCEAUX"

# ─── Rover attitude display (GY-80 IMU) ───────────────────────
# Masked orthographic renders of the rover, one per axis view. Missing
# files fall back to a built-in SVG silhouette, so a partial set is fine.
#   top.png   -> yaw     front.png -> roll     side.png  -> pitch
ROVER_VIEW_DIR        = ROOT_DIR / "src" / "dashboard" / "static" / "rover"
ROVER_VIEW_EXTENSIONS = (".png", ".webp", ".svg", ".jpg")

# ─── Ge'ez LLM restoration (fine-tuning) ───────────────────────
GEEZ_RESTORATION_CORPUS   = DATA_DIR / "corpus" / "geez_inscriptions.jsonl"
GEEZ_RESTORATION_TRAIN    = DATA_DIR / "geez_restoration_train.jsonl"
GEEZ_RESTORATION_VAL      = DATA_DIR / "geez_restoration_val.jsonl"
GEEZ_RESTORATION_EVAL     = DATA_DIR / "geez_restoration_eval.jsonl"
GEEZ_RESTORATION_LEGACY   = DATA_DIR / "geez_restoration_dataset.jsonl"
HHD_MERGED_CSV            = DATA_DIR / "geez_merged" / "train_raw" / "image_text_pairs_train.csv"
HHD_FALLBACK_CSV          = DATA_DIR / "geez_characters" / "train_raw" / "image_text_pairs_train.csv"
RESTORATION_EXAMPLES_PER_PHRASE = 12
RESTORATION_DAMAGE_RATES  = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
RESTORATION_TRAIN_RATIO   = 0.80
RESTORATION_VAL_RATIO     = 0.10
RESTORATION_HHD_MAX_PHRASES   = 2500
RESTORATION_HHD_MIN_GRAPHEMES = 3
RESTORATION_HHD_MAX_GRAPHEMES = 80
RESTORATION_SPLIT_SEED    = 42
OCR_CHAR_CONF_THRESHOLD   = 0.45   # per-frame max prob below → [MISSING]
OCR_USE_MISSING_TOKENS      = True   # inject [MISSING] in predict() when low conf
LM_STUDIO_BASE_URL        = "http://127.0.0.1:1234/v1"  # LM Studio OpenAI-compatible API
LM_STUDIO_TIMEOUT_SEC     = 30.0   # HTTP timeout for model list + chat completions

# ─── Database ─────────────────────────────────────────────────
DB_PATH             = DATA_DIR / "databases" / "axum_heritage.db"
INSCRIPTIONS_JSON   = DATA_DIR / "databases" / "inscriptions.json"
ARTEFACTS_JSON      = DATA_DIR / "databases" / "artefacts.json"
CONSERVATION_KB_PATH = DATA_DIR / "databases" / "conservation_kb.json"

# ─── Treatment Advisor ────────────────────────────────────────
# Maps artefact classifier labels → default substrate in conservation KB
ARTEFACT_SUBSTRATE_MAP = {
    "pottery":               "terracotta_ceramic",
    "stone_carving":         "basalt",
    "coin":                  "metal_bronze",
    "inscription_fragment":  "limestone_porous",
    "other":                 "limestone_porous",
}
TREATMENT_URGENCY_CRITICAL_YEARS = 5    # fragility clock below → emergency
TREATMENT_URGENCY_PRIORITY_YEARS  = 20   # fragility clock below → priority

# ─── Fragility Clock ──────────────────────────────────────────
# Weights sum to 1.0 -- relative importance of each damage dimension in the
# composite score. Crack + salt dominate (structural + chemical decay are
# the two fastest failure modes); stress/moisture/bio contribute but don't
# dominate on their own. Reasoned starting point, not fitted to field data
# -- see fragility_clock.calibrate_baseline() for recalibrating against
# real cases once available.
FRAGILITY_WEIGHT_CRACK    = 0.35
FRAGILITY_WEIGHT_SALT     = 0.30
FRAGILITY_WEIGHT_STRESS   = 0.15
FRAGILITY_WEIGHT_MOISTURE = 0.10
FRAGILITY_WEIGHT_BIO      = 0.10

# Substrate baseline years (time-to-loss at zero measured damage).
# Literature-motivated relative ordering (stone > metal > porous stone >
# ceramic under equal damage), not measured against real AXUM finds.
# Keys must match ARTEFACT_SUBSTRATE_MAP's values above, plus 'default'
# for resolve_substrate()'s fallback.
FRAGILITY_BASELINE_YEARS = {
    "basalt":              300,   # dense igneous stone -- most durable
    "metal_bronze":        200,   # corrodes slowly if not actively wet
    "limestone_porous":    100,   # porous -- vulnerable to salt/moisture cycling
    "terracotta_ceramic":   80,   # fired clay -- fastest degrading of the four
    "default":             100,
}

# ─── Fragment Grouping ────────────────────────────────────────
FRAGMENT_MATCH_POSSIBLE_MIN  = 0.65   # possible same-vessel match
FRAGMENT_MATCH_CONFIRMED_MIN = 0.85   # confirmed fragment pair
FRAGMENT_ICP_MAX_DISTANCE_M  = 0.005  # 5 mm ICP correspondence threshold
FRAGMENT_DENSITY_TOLERANCE   = 0.5    # g/cm³ — beyond this, reject match

# ── LED ────────────────────────────────────────────────────────
# NeoPixel ring was dropped from the build. Real hardware as of this
# integration pass: 4x quadrant LEDs (photometric stereo, N/E/S/W) + 1x UV
# LED, driven via PCA9685 + MOSFETs on the Arduino side (see
# arduino/axum_rover/axum_rover.ino, LED:QUAD:*/LED:UV:* commands, and
# src/arm/controller.py's LightingController). No Python-side pin/count
# constants needed -- that's firmware-side now via I2C channel numbers.

# ─── Dashboard ────────────────────────────────────────────────
DASHBOARD_HOST  = "0.0.0.0"
DASHBOARD_PORT  = 5000



# ── Logging ────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"          # DEBUG, INFO, WARNING, ERROR
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "axum_rover.log"

# ── Multispectral (NDCI stress mapping) ──────────────────────────
# Added in this integration pass — multispectral.py imports these but they
# didn't exist anywhere in config.py, so the module could not be imported
# at all until now. Values below are reasoned starting points (not
# measured against real visible/IR captures) -- needs a calibration pass
# once real quadrant-LED + IR frame pairs exist (see
# calibrate_ndci_baseline() in multispectral.py, built for exactly this).
MULTISPECTRAL_EPSILON                 = 1e-6   # avoid div-by-zero in NDCI ratio
MULTISPECTRAL_MIN_REGION_AREA_PX      = 30     # matches MIN_CRACK_AREA -- same
                                                 # contours, just re-scored
MULTISPECTRAL_HAZARD_SCORE_MIN        = 0.40   # >= this -> "hazard"
MULTISPECTRAL_SURFACE_ONLY_SCORE_MAX  = 0.15   # <= this -> "surface-only"
                                                 # (between the two = "monitor")

# ── Photometric stereo (relief mapping) ──────────────────────────
# Same situation as above -- photometric_stereo.py imports these, none
# existed. Values are reasoned defaults, not measured.
PHOTOMETRIC_ALBEDO_EPSILON            = 1e-6   # avoid div-by-zero normalizing normals
PHOTOMETRIC_DEPTH_GRADIENT_SCALE      = 1.0    # neutral scale on p/q gradients
PHOTOMETRIC_DEPTH_SMOOTH_ITERATIONS   = 30     # relief integration smoothing passes

# ── Compute tier (CPU/GPU) ────────────────────────────────────────
# Added this pass -- despite being referenced as "already defined
# elsewhere" by two separate handoffs (Robotics Software Engineer's
# earlier work, AI & Computer Vision Engineer's GPU-tier handoff), it did
# not actually exist anywhere in this file. Confirmed by direct search
# before adding, not assumed. "cpu" is the safe default -- GPU-tier code
# paths (src/object_detection/device_utils.py) fall back to CPU weights
# with a loud warning if "gpu" is requested but no CUDA device is
# actually present, so this is safe to leave as "cpu" on any machine.
COMPUTE_TIER = "cpu"   # "cpu" or "gpu" -- set to "gpu" on the GPU-tier test machine

# GPU-tier restoration model name. None until ML Research Engineer's
# ablation (which model size actually wins for restoration) names a
# specific larger variant -- until then, GPU tier requests the SAME
# model as CPU tier, just asking for GPU acceleration. Actual GPU
# offload is a manual LM Studio server-side setting, NOT controlled by
# this value or any Python code -- see src/ocr/llm_restoration.py's
# _resolve_restoration_model_name() for the full explanation.
RESTORATION_MODEL_GPU = None   # e.g. "qwen2.5-7b-instruct" once ablation confirms
