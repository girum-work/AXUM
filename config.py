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
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FPS      = 10

# ─── Crack Detection ──────────────────────────────────────────
BLUR_KERNEL     = (5, 5)
USE_AUTO_CANNY  = True
CANNY_T1        = 50
CANNY_T2        = 150
MIN_CRACK_AREA  = 30
MIN_CRACK_LEN   = 40
MIN_ASPECT_RATIO= 3.0
CRACK_SEVERITY_THRESHOLD = 0.4  # below = treatable, above = flag

# ─── Heatmap ──────────────────────────────────────────────────
GRID_ROWS       = 10
GRID_COLS       = 15
WALL_WIDTH_CM   = 80.0
ROVER_SPEED_CMS = 10.0

# ─── OCR ──────────────────────────────────────────────────────
# NOTE: OCR_IMG_SIZE and NUM_GEEZ_CLASSES appear twice below — the SECOND
# assignment wins in Python. The (32, 128) and 300 values are the active ones.
OCR_MODEL_PATH      = MODELS_DIR / "geez_ocr.pth"
OCR_CONFIDENCE_MIN  = 0.50    # below this = flag for expert review
OCR_IMG_SIZE        = 64      # DEPRECATED — overridden by (32, 128) on next line
NUM_GEEZ_CLASSES    = 231     # DEPRECATED — overridden by 300 below
OCR_IMG_SIZE      = (32, 128)   # height, width in pixels — active OCR input shape
NUM_GEEZ_CLASSES  = 300         # active charset size after build_geez_charset()
OCR_BEAM_WIDTH    = 5           # CTC beam search width at inference
OCR_USE_BEAM_DECODE = True      # beam search vs greedy (no retrain needed)
OCR_USE_WEIGHTED_SAMPLER = True # oversample rare Ge'ez characters in training
OCR_USE_STONE_AUGMENT = True    # albumentations stone-inscription train augment
OCR_USE_ADAPTIVE_BINARIZE = True  # CLAHE + adaptive threshold in preprocess_for_ocr

# ── Models ─────────────────────────────────────────────────────
OCR_MODEL_PATH  = MODELS_DIR / "geez_ocr.pth"
OBJ_MODEL_PATH  = MODELS_DIR / "artefact_classifier.pth"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8_artefacts.pt"
                 # Set to None to use COCO fallback before fine-tuning

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
YOLO_COIN_MODEL_PATH = MODELS_DIR / "yolov8_coins.pt"
# YOLO class names for era + denomination (train_yolov8_coins.py)
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

# ─── Photogrammetry ───────────────────────────────────────────
MESHROOM_PATH        = r"C:\Meshroom\Meshroom.exe"   # update after install
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
MESH_TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

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

# ─── Fragment Grouping ────────────────────────────────────────
FRAGMENT_MATCH_POSSIBLE_MIN  = 0.65   # possible same-vessel match
FRAGMENT_MATCH_CONFIRMED_MIN = 0.85   # confirmed fragment pair
FRAGMENT_ICP_MAX_DISTANCE_M  = 0.005  # 5 mm ICP correspondence threshold
FRAGMENT_DENSITY_TOLERANCE   = 0.5    # g/cm³ — beyond this, reject match

# ── LED ────────────────────────────────────────────────────────
LED_PIN       = 11
LED_COUNT     = 12           # number of NeoPixels in your ring
LED_BRIGHTNESS= 0.6          # 0.0 to 1.0

# ─── Dashboard ────────────────────────────────────────────────
DASHBOARD_HOST  = "0.0.0.0"
DASHBOARD_PORT  = 5000



# ── Logging ────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"          # DEBUG, INFO, WARNING, ERROR
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "axum_rover.log"