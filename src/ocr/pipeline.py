# src/ocr/pipeline.py
"""
AXUM ROVER — Ge'ez OCR Pipeline
=================================
Dataset preparation, image preprocessing, model training,
and full inference pipeline for Ge'ez text recognition.

This file handles everything AROUND the model:
- Preprocessing camera images before feeding to the model
- Loading and augmenting the HHD-Ethiopic training dataset
- Training loop with CTC loss
- Text region detection (finding WHERE text is before reading it)
- Post-processing: translation lookup and confidence display

Author: Axum Rover Team
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.object_detection.device_utils import resolve_device
from config import (
    COMPUTE_TIER,
    DATA_DIR,
    INSCRIPTIONS_JSON,
    NUM_GEEZ_CLASSES,
    OCR_CONFIDENCE_MIN,
    OCR_IMG_SIZE,
    OCR_MODEL_PATH,
    OCR_USE_ADAPTIVE_BINARIZE,
    OCR_USE_BEAM_DECODE,
    OCR_USE_STONE_AUGMENT,
    OCR_USE_WEIGHTED_SAMPLER,
    OCR_CTC_SEQ_LEN,
)
from src.ocr.training_contract import (
    charset_fingerprint,
    compute_sequence_metrics,
    label_fits_ctc,
    min_ctc_timesteps,
    normalize_ocr_label,
)
from src.ocr.model import (
    BLANK_IDX,
    CHAR_TO_IDX,
    GEEZ_CHARSET,
    IDX_TO_CHAR,
    GeezOCRModel,
    load_ocr_model,
    save_ocr_model,
)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — IMAGE PREPROCESSING FOR OCR
# ═══════════════════════════════════════════════════════════════


def resize_pad_image(
    image: np.ndarray,
    target_size: tuple[int, int] = OCR_IMG_SIZE,
    fill: int = 255,
) -> np.ndarray:
    """Resize without distortion and center on a fixed OCR canvas."""
    target_height, target_width = target_size
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError(f"invalid OCR image shape: {image.shape}")
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * scale)))
    resized_height = max(1, min(target_height, round(height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas_shape = (target_height, target_width, *image.shape[2:])
    canvas = np.full(canvas_shape, fill, dtype=image.dtype)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_for_ocr(
    image: np.ndarray,
    use_adaptive_binarize: bool = None,
) -> torch.Tensor:
    """
    Preprocess an image of a text region for OCR model input.

    The model expects:
    - Grayscale (single channel)
    - Size: 32 pixels tall × 128 pixels wide
    - Normalized to mean=0.5, std=0.5 (maps 0-255 to roughly -1 to 1)
    - Shape: (1, 1, 32, 128) — batch of 1, 1 channel

    Stone inscription preprocessing (Fix 4):
        CLAHE → adaptive binarization → morphological closing reconnects
        eroded Ge'ez strokes. When ``use_adaptive_binarize`` is False,
        falls back to bilateral denoise + light closing for parchment scans.

    Args:
        image: BGR or grayscale image containing ONE text region
        use_adaptive_binarize: Override OCR_USE_ADAPTIVE_BINARIZE from config

    Returns:
        tensor: Shape (1, 1, 32, 128), normalized float32
    """
    if use_adaptive_binarize is None:
        use_adaptive_binarize = OCR_USE_ADAPTIVE_BINARIZE

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    if use_adaptive_binarize:
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    resized = resize_pad_image(processed)

    pil_img = Image.fromarray(resized)

    # Step 6: Normalize to [-1, 1] range
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # → [0, 1] float tensor
            transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
        ]
    )

    tensor = transform(pil_img)  # Shape: (1, 32, 128)
    tensor = tensor.unsqueeze(0)  # Shape: (1, 1, 32, 128)

    return tensor


def enhance_inscription_image(image: np.ndarray) -> np.ndarray:
    """
    Apply specialized preprocessing for stone inscription images.

    Stone inscriptions have unique visual characteristics:
    - Characters are carved INTO the surface (dark grooves)
    - Background is uneven stone texture
    - Lighting creates shadows that both help and hinder detection

    This pipeline enhances the carved character edges specifically.

    Args:
        image: BGR image of a stone surface with inscriptions

    Returns:
        enhanced: Grayscale image with inscription features enhanced
    """
    # Convert to grayscale
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )

    # Step 1: Normalize illumination with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    # Step 2: Unsharp masking — sharpen edges
    # Subtract a blurred version to enhance high-frequency details (edges)
    blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(normalized, 1.5, blurred, -0.5, 0)

    # Step 3: Adaptive threshold to binarize carved characters
    # Carved characters appear as dark regions on lighter stone background
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # INV: characters become white
        blockSize=25,
        C=8,
    )

    # Step 4: Remove small noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


# ═══════════════════════════════════════════════════════════════
# SECTION 2b — STONE INSCRIPTION AUGMENTATION (albumentations)
# ═══════════════════════════════════════════════════════════════


def build_stone_inscription_transform():
    """
    Build albumentations pipeline simulating stone carving domain shift.

    WHAT: Applies emboss, noise, erosion/dilation, and perspective transforms
    that mimic real stone inscription capture conditions.
    WHY: HHD-Ethiopic is handwritten parchment — without domain augmentation
    the model underperforms on carved stone surfaces at demo time.

    Returns:
        albumentations.Compose transform, or None if albumentations unavailable.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import inspect
    except ImportError as exc:
        logger.warning(f"albumentations unavailable ({exc}); using torchvision aug")
        return None

    def _shift_scale_rotate():
        """ShiftScaleRotate with version-compatible padding parameter."""
        kwargs = dict(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=8,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.6,
        )
        params = inspect.signature(A.ShiftScaleRotate.__init__).parameters
        if "fill" in params:
            return A.ShiftScaleRotate(**kwargs, fill=255)
        return A.ShiftScaleRotate(**kwargs, value=255)

    def _elastic_transform():
        """ElasticTransform — alpha_affine removed in albumentations 1.4+."""
        params = inspect.signature(A.ElasticTransform.__init__).parameters
        if "alpha_affine" in params:
            kwargs = dict(alpha=30, sigma=5, alpha_affine=5, border_mode=0, p=0.4)
            try:
                return A.ElasticTransform(**kwargs, value=255)
            except TypeError:
                return A.ElasticTransform(**kwargs, fill=255)
        return A.ElasticTransform(alpha=30, sigma=5, p=0.4)

    def _gauss_noise():
        """GaussNoise — API varies across albumentations versions."""
        params = inspect.signature(A.GaussNoise.__init__).parameters
        if "std_range" in params:
            return A.GaussNoise(std_range=(0.04, 0.2), p=1.0)
        if "noise_scale" in params:
            return A.GaussNoise(noise_scale=(0.04, 0.2), p=1.0)
        return A.GaussNoise(var_limit=(10, 50), p=1.0)

    def _morphological_wear():
        """Erosion/dilation — A.Erosion removed in albumentations 1.4+."""
        if hasattr(A, "Erosion"):
            return A.OneOf(
                [
                    A.Erosion(scale=(1, 2), p=1.0),
                    A.Dilation(scale=(1, 2), p=1.0),
                ],
                p=0.4,
            )
        return A.OneOf(
            [
                A.Morphological(scale=(1, 2), operation="erosion", p=1.0),
                A.Morphological(scale=(1, 2), operation="dilation", p=1.0),
            ],
            p=0.4,
        )

    return A.Compose(
        [
            A.Lambda(image=lambda image, **_kwargs: resize_pad_image(image)),
            A.OneOf(
                [
                    A.Emboss(alpha=(0.4, 0.8), strength=(0.4, 0.8), p=1.0),
                    A.Sharpen(alpha=(0.3, 0.7), lightness=(0.8, 1.2), p=1.0),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    _gauss_noise(),
                    A.ISONoise(
                        color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0
                    ),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ],
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.4),
                contrast_limit=(-0.3, 0.5),
                p=0.8,
            ),
            _morphological_wear(),
            _shift_scale_rotate(),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            _elastic_transform(),
            A.ToGray(num_output_channels=1, p=1.0),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.8),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


# Module-level transform built lazily on first training dataset init
stone_inscription_transform = None


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — TEXT REGION DETECTION
# ═══════════════════════════════════════════════════════════════


def detect_text_regions(
    image: np.ndarray,
    min_width: int = 30,
    min_height: int = 20,
    max_width: int = None,
    max_height: int = None,
) -> list:
    """
    Find regions in an image that likely contain Ge'ez text.

    This is the step BEFORE OCR — we need to find WHERE the text is
    before we can read it. On a stone artefact, text could be anywhere.

    Method: Morphological text detection
        1. Enhance inscription features (CLAHE + threshold)
        2. Dilate horizontally to connect characters within a word
        3. Find contours of connected character groups (= text lines)
        4. Filter by size (too small = noise, too large = decoration)
        5. Return bounding boxes with padding

    Why not use a neural text detector (like EAST or CRAFT)?
        Those models require ~100MB and run at 1-2fps on CPU.
        For the artefact scanning workflow, speed is less critical
        (we're scanning still images, not video), but we'd need to
        download and integrate a separate model.
        The morphological approach runs in <10ms and is sufficient
        for well-lit close-up inscription images.

    Args:
        image: BGR image of an artefact surface
        min_width:  Minimum text region width in pixels
        min_height: Minimum text region height in pixels
        max_width:  Maximum text region width (None = no limit)
        max_height: Maximum text region height (None = no limit)

    Returns:
        regions: List of dicts:
            {
                'bbox': (x, y, w, h),    # bounding box with padding
                'crop': np.ndarray,       # cropped image of this region
                'confidence': float       # detection confidence 0-1
            }
    """
    h_img, w_img = image.shape[:2]
    if max_width is None:
        max_width = w_img
    if max_height is None:
        max_height = h_img // 2

    # Enhance inscription features
    enhanced = enhance_inscription_image(image)

    # Dilate horizontally to connect characters in the same word/line
    # A wide, flat kernel connects nearby characters horizontally
    # but not vertically (prevents merging multiple text lines)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(enhanced, h_kernel, iterations=1)

    # Find contours of connected text groups
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Apply size filters
        if w < min_width or h < min_height:
            continue
        if w > max_width or h > max_height:
            continue

        # Aspect ratio filter: text lines are wider than tall
        if w / h < 1.5:
            continue

        # Add padding around the detected region (improves OCR accuracy)
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        # Crop the original image (not the enhanced version)
        # The model was trained on natural images, not binary
        crop = image[y1:y2, x1:x2]

        # Confidence: based on fill ratio (dense text = high confidence)
        region_mask = enhanced[y : y + h, x : x + w]
        fill_ratio = np.sum(region_mask > 0) / (w * h)
        # Text fill ratio typically 0.2-0.6; noise is usually <0.1 or >0.8
        confidence = min(fill_ratio * 2.5, 1.0) if 0.1 < fill_ratio < 0.8 else 0.3

        regions.append(
            {
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "crop": crop,
                "confidence": confidence,
                "fill_ratio": fill_ratio,
            }
        )

    # Sort top-to-bottom, then left-to-right (natural reading order)
    regions.sort(key=lambda r: (r["bbox"][1] // 30, r["bbox"][0]))

    logger.debug(f"Text regions detected: {len(regions)}")
    return regions


def draw_text_region_overlay(
    image: np.ndarray, regions: list, ocr_results: list = None
) -> np.ndarray:
    """
    Draw text region bounding boxes and OCR results on the image.

    Args:
        image: Original BGR image
        regions: From detect_text_regions()
        ocr_results: Optional list of OCR prediction dicts

    Returns:
        overlay: Annotated image
    """
    overlay = image.copy()

    for i, region in enumerate(regions):
        x, y, w, h = region["bbox"]

        # Color by confidence: green (high) → orange (low)
        conf = region["confidence"]
        color = (0, int(255 * conf), int(255 * (1 - conf)))

        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Region index
        cv2.putText(
            overlay,
            f"T{i + 1}",
            (x + 4, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # OCR result if available
        if ocr_results and i < len(ocr_results):
            result = ocr_results[i]
            text = result.get("text", "")
            oconf = result.get("confidence", 0)

            if text:
                label = f"{text} ({oconf:.0%})"
                # White background for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(overlay, (x, y - th - 10), (x + tw + 4, y), (0, 0, 0), -1)
                cv2.putText(
                    overlay,
                    label,
                    (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0) if result.get("reliable") else (0, 165, 255),
                    1,
                )

    return overlay


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATASET: HHD-ETHIOPIC
# ═══════════════════════════════════════════════════════════════


class HHDEthiopicDataset(Dataset):
    """
    Dataset loader for the HHD-Ethiopic (Historical Handwritten Document)
    dataset from HuggingFace.

    Dataset: OCR-Ethiopic/HHD-Ethiopic
    URL: https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic

    Dataset structure (after download):
        data/geez_characters/
            ሀ/          ← folder name IS the character
                001.png
                002.png
                ...
            ሁ/
                ...
            ... (one folder per character)

    If using the HuggingFace datasets library format:
        The dataset contains:
        - 'image': PIL Image of the character/word
        - 'text': Ground truth Ge'ez text string
        - 'source': Source document identifier

    We support both formats: folder-based and HuggingFace streaming.

    Augmentation strategy for small datasets:
        With only 50-200 samples per character, overfitting is a major risk.
        We apply aggressive augmentation to simulate the variations seen
        in real stone inscriptions:
        - Rotation: inscriptions are rarely perfectly level
        - Perspective distortion: camera angle varies
        - Elastic distortion: carved characters have irregular edges
        - Noise: stone texture adds visual noise
        - Brightness/contrast: lighting variation from LED ring
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples_per_class: int = 500,
        augment: bool = True,
        use_stone_augment: bool = None,
        manifest_path: str | Path | None = None,
        image_dir: str | Path | None = None,
    ):
        """
        Args:
            data_dir: Path to data/geez_characters/ folder
            split: 'train', 'val', or 'test'
            max_samples_per_class: Cap per character class (prevents imbalance)
            augment: Apply data augmentation (True for training only)
            use_stone_augment: Albumentations stone pipeline (default: config flag)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.samples = []
        self.albu_transform = None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest_image_dir = Path(image_dir) if image_dir else None

        if use_stone_augment is None:
            use_stone_augment = OCR_USE_STONE_AUGMENT

        if self.augment and use_stone_augment:
            global stone_inscription_transform
            if stone_inscription_transform is None:
                stone_inscription_transform = build_stone_inscription_transform()
            self.albu_transform = stone_inscription_transform

        self.transform = self._build_transform()

        self._load_samples(max_samples_per_class)

        logger.info(f"HHD-Ethiopic dataset ({split}): {len(self.samples)} samples")

    def _build_transform(self) -> transforms.Compose:
        """Build transform pipeline based on split (train/val)."""
        if self.augment and self.albu_transform is not None:
            return None

        if self.augment:
            return transforms.Compose(
                [
                    transforms.Lambda(
                        lambda image: Image.fromarray(
                            resize_pad_image(np.array(image), OCR_IMG_SIZE)
                        )
                    ),
                    transforms.RandomRotation(
                        degrees=8,
                        fill=255,  # fill with white
                    ),
                    transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Lambda(
                        lambda image: Image.fromarray(
                            resize_pad_image(np.array(image), OCR_IMG_SIZE)
                        )
                    ),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def _load_samples(self, max_per_class: int):
        """
        Load sample paths from the data directory.

        Supports two organization styles:
        Style A: data/geez_characters/CHARACTER/image.png
        Style B: data/geez_characters/train_raw/image_text_pairs_train.csv
                 with images in train_raw/image_train/
        """
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return

        import csv

        if self.manifest_path is not None:
            if not self.manifest_path.exists():
                logger.error(f"OCR manifest not found: {self.manifest_path}")
                return
            image_dir = self.manifest_image_dir or self.manifest_path.parent
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            rejected = 0
            for row in rows:
                if len(row) < 2:
                    rejected += 1
                    continue
                image_name = row[0].strip()
                text = normalize_ocr_label(row[1])
                image_path = image_dir / image_name
                if (
                    not image_name
                    or not image_path.exists()
                    or not label_fits_ctc(text, OCR_CTC_SEQ_LEN)
                    or any(character not in CHAR_TO_IDX for character in text)
                ):
                    rejected += 1
                    continue
                self.samples.append((image_path, text))
            logger.info(
                f"Explicit OCR manifest loaded: {len(self.samples)} samples, "
                f"{rejected} rejected ({self.manifest_path})"
            )
            return

        # Style B: CSV-based dataset — prefer when any subdir has the pairs file
        csv_subset_dir = None
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and (subdir / "image_text_pairs_train.csv").exists():
                csv_subset_dir = subdir
                break

        if csv_subset_dir is not None:
            csv_path = csv_subset_dir / "image_text_pairs_train.csv"
            img_dir = csv_subset_dir / "image_train"

            if not img_dir.exists():
                logger.error(f"Image dir missing in {csv_subset_dir}")
                return

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

            rng = random.Random(42)  # fixed seed — identical order for train/val calls, no leakage
            rng.shuffle(rows)
            split_idx = int(len(rows) * 0.85)
            if self.split == "train":
                rows = rows[:split_idx]
            else:
                rows = rows[split_idx:]

            for row in rows:
                if len(row) < 2:
                    continue
                img_name = row[0].strip()
                text = normalize_ocr_label(row[1])
                if not text:
                    continue
                if not label_fits_ctc(text, OCR_CTC_SEQ_LEN):
                    continue
                if any(character not in CHAR_TO_IDX for character in text):
                    continue
                img_path = img_dir / img_name
                if img_path.exists():
                    self.samples.append((img_path, text))

            logger.info(
                f"CSV loaded: {len(self.samples)} samples ({self.split})"
            )
            return

        # Style A: character folders
        char_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for char_dir in char_dirs:
            char_name = char_dir.name

            # Verify this is a valid Ge'ez character
            if char_name not in CHAR_TO_IDX:
                # Try to match folder name to a known character
                # (some datasets use Unicode code point as folder name)
                try:
                    if char_name.startswith("U+") or char_name.startswith("0x"):
                        cp = int(char_name.replace("U+", "").replace("0x", ""), 16)
                        char = chr(cp)
                        if char in CHAR_TO_IDX:
                            char_name = char
                        else:
                            continue
                    else:
                        continue
                except ValueError:
                    continue

            # Load images from this character's folder
            images = (
                list(char_dir.glob("*.png"))
                + list(char_dir.glob("*.jpg"))
                + list(char_dir.glob("*.jpeg"))
            )

            # Split train/val 85%/15% (shuffled — fixed seed, matches CSV-path split logic)
            rng = random.Random(42)
            rng.shuffle(images)
            split_idx = int(len(images) * 0.85)
            if self.split == "train":
                images = images[:split_idx]
            else:
                images = images[split_idx:]

            # Cap per class
            images = images[:max_per_class]

            for img_path in images:
                self.samples.append((img_path, char_name))

    def __len__(self):
        return len(self.samples)

    def _imbalance_key(self, label: str, char_counts: dict) -> str:
        """
        Key for class-balanced sampling.

        Single-character folders use that character. Multi-character CSV
        labels use the rarest Ge'ez glyph in the string so lines with
        uncommon syllables are seen more often during training.
        """
        chars = [c for c in label if c in CHAR_TO_IDX]
        if len(chars) == 1:
            return chars[0]
        if chars:
            return min(chars, key=lambda c: char_counts.get(c, 1))
        return label if label else "<UNK>"

    def get_class_weights(self) -> list:
        """
        Per-sample weights for WeightedRandomSampler.

        Rare characters (or samples containing them) get higher weight so
        the model is not dominated by high-frequency syllables like ሰ, ማ.

        Returns:
            List of float weights, one per sample in self.samples.
        """
        from collections import Counter

        char_counts = Counter()
        for _, label in self.samples:
            for c in label:
                if c in CHAR_TO_IDX:
                    char_counts[c] += 1

        sample_keys = [self._imbalance_key(label, char_counts) for _, label in self.samples]
        key_counts = Counter(sample_keys)
        weights = [1.0 / key_counts[k] for k in sample_keys]
        return weights

    def __getitem__(self, idx: int) -> tuple:
        """
        Load one sample.

        Returns:
            image_tensor: Shape (1, 32, 128) normalized float32
            label_indices: List of integer character indices (for CTC)
            label_length: Integer — number of characters in label
        """
        img_path, text = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (128, 32), color=255)

        # Apply transforms (albumentations for stone domain, else torchvision)
        if self.albu_transform is not None:
            arr = np.array(image.convert("RGB"))
            augmented = self.albu_transform(image=arr)
            tensor = augmented["image"]
            if tensor.ndim == 3 and tensor.shape[0] == 3:
                tensor = tensor.mean(dim=0, keepdim=True)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = self.transform(image)

        # Convert text label to integer indices for CTC
        label_indices = []
        for char in text:
            idx_val = CHAR_TO_IDX.get(char, CHAR_TO_IDX.get("<UNK>", 1))
            label_indices.append(idx_val)

        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        label_length = torch.tensor(len(label_indices), dtype=torch.long)

        return tensor, label_tensor, label_length


def ctc_collate_fn(batch: list) -> tuple:
    """
    Custom collate function for DataLoader with CTC loss.

    CTC requires:
    1. All images in a batch have the SAME size (padded if needed) ✓
    2. Labels are concatenated into a 1D tensor
    3. Label lengths are provided separately (to un-concatenate)
    4. Input lengths are provided (sequence length for each image)

    Args:
        batch: List of (image_tensor, label_tensor, label_length)

    Returns:
        images:         (batch_size, 1, 32, 128)
        labels:         (sum_of_all_label_lengths,) — concatenated
        input_lengths:  (batch_size,) — sequence length for each image
        label_lengths:  (batch_size,) — label length for each sample
    """
    images, labels_list, label_lengths = zip(*batch)

    images = torch.stack(images, 0)
    labels = torch.cat(labels_list, 0)

    input_lengths = torch.full(
        (len(images),), fill_value=OCR_CTC_SEQ_LEN, dtype=torch.long
    )

    label_lengths = torch.stack([l.clone().detach() for l in label_lengths]).long()

    offset = 0
    for length in label_lengths.tolist():
        sequence = labels[offset : offset + length].tolist()
        required = length + sum(left == right for left, right in zip(sequence, sequence[1:]))
        if required > OCR_CTC_SEQ_LEN:
            raise ValueError(
                f"CTC target requires {required} steps but model provides {OCR_CTC_SEQ_LEN}"
            )
        offset += length

    return images, labels, input_lengths, label_lengths


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DOWNLOAD HHD-ETHIOPIC DATASET
# ═══════════════════════════════════════════════════════════════


def download_hhd_ethiopic(save_dir: str):
    """
    Download the HHD-Ethiopic dataset from HuggingFace.

    Prerequisites:
        pip install datasets huggingface_hub

    The dataset will be saved in the folder structure expected
    by HHDEthiopicDataset:
        save_dir/CHARACTER/image_XXX.png

    This function downloads and organizes the dataset automatically.
    Run it ONCE before training. It downloads ~500MB.

    Args:
        save_dir: Where to save — use data/geez_characters/
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading HHD-Ethiopic dataset from HuggingFace...")
    logger.info("This will download ~500MB. Do not interrupt.")

    try:
        # Try using datasets library
        from datasets import load_dataset

        dataset = load_dataset("OCR-Ethiopic/HHD-Ethiopic", trust_remote_code=True)

        logger.info(f"Dataset loaded: {dataset}")

        # Process each split
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            logger.info(f"Processing {split_name}: {len(split_data)} samples")

            for i, sample in enumerate(split_data):
                # Get the image and label
                image = sample.get("image")
                text = sample.get("text", sample.get("label", ""))

                if image is None or not text:
                    continue

                # Use first character as folder name
                # (handles both single-char and word-level labels)
                char = text[0] if text else "unknown"

                # Create character folder
                char_dir = save_path / char
                char_dir.mkdir(exist_ok=True)

                # Save image
                img_filename = f"{split_name}_{i:06d}.png"
                img_path = char_dir / img_filename

                if hasattr(image, "save"):
                    image.save(img_path)
                else:
                    import numpy as np

                    cv2.imwrite(str(img_path), np.array(image))

                if i % 1000 == 0:
                    logger.info(f"  Saved {i} samples...")

        logger.info(f"Dataset saved to: {save_path}")
        logger.info("Run verify_dataset() to check the download.")

    except ImportError:
        logger.error("datasets library not installed.")
        logger.error("Run: pip install datasets huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error(
            "Alternative: manually download from "
            "https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic"
        )
        raise


def verify_dataset(data_dir: str) -> dict:
    """
    Check the dataset structure and report statistics.

    Run this after download_hhd_ethiopic() to confirm the dataset
    was saved correctly before starting training.

    Args:
        data_dir: Path to data/geez_characters/

    Returns:
        stats: Dict with dataset statistics
    """
    data_path = Path(data_dir)
    char_counts = {}
    total = 0

    if not data_path.exists():
        logger.error(f"Directory not found: {data_dir}")
        return {}

    for char_dir in sorted(data_path.iterdir()):
        if not char_dir.is_dir():
            continue

        count = len(list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg")))

        if count > 0:
            char_counts[char_dir.name] = count
            total += count

    stats = {
        "total_images": total,
        "total_classes": len(char_counts),
        "min_per_class": min(char_counts.values()) if char_counts else 0,
        "max_per_class": max(char_counts.values()) if char_counts else 0,
        "avg_per_class": total / len(char_counts) if char_counts else 0,
        "char_counts": char_counts,
    }

    print("\n" + "=" * 50)
    print("DATASET VERIFICATION REPORT")
    print("=" * 50)
    print(f"Total images:     {stats['total_images']:,}")
    print(f"Total classes:    {stats['total_classes']}")
    print(f"Min per class:    {stats['min_per_class']}")
    print(f"Max per class:    {stats['max_per_class']}")
    print(f"Avg per class:    {stats['avg_per_class']:.1f}")
    print(f"Classes < 20 img: {sum(1 for c in char_counts.values() if c < 20)}")
    print("=" * 50)

    if stats["total_images"] < 1000:
        print("\nWARNING: Dataset is small. Consider:")
        print("  1. Adding more images per character (target: 50+ each)")
        print("  2. Using heavier augmentation (already configured)")
        print("  3. Pre-training on a larger Ethiopic dataset first")

    return stats


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — OCR TRAINING LOOP
# ═══════════════════════════════════════════════════════════════


def _save_resume_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: "GeezOCRModel",
    optimizer,
    scheduler,
    best_val_loss: float,
    patience_counter: int,
    epoch_metrics: list,
) -> None:
    """
    WHAT: saves FULL training state (model, optimizer, scheduler, epoch,
    early-stopping counters, metric history) for exact-resume, not just
    model weights.
    WHY separate from save_ocr_model(): save_ocr_model() saves the BEST
    model's weights only, for deployment/inference — that behavior is
    unchanged. This checkpoint is for exact training-state resume after an
    interrupted session (e.g. a Kaggle kernel dying mid-run) and needs the
    optimizer/scheduler state too, or resuming would silently NOT reproduce
    the same training trajectory.
    WHY atomic write (tmp file + rename): a checkpoint saved every single
    epoch is a real risk surface — if the process dies mid-write (exactly
    the kind of interruption this exists to protect against), a partial
    file must never become the thing a resume attempt tries to load.
    Writing to a .tmp path first and renaming only after a full successful
    write means the real checkpoint path is always either the last
    complete checkpoint or absent, never corrupted.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")

    cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        "epoch": epoch,
        "model_state_dict": cpu_model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "epoch_metrics": epoch_metrics,
        "charset_fingerprint": charset_fingerprint(GEEZ_CHARSET),
        "image_size": list(OCR_IMG_SIZE),
        "ctc_seq_len": OCR_CTC_SEQ_LEN,
    }, tmp_path)

    tmp_path.replace(checkpoint_path)  # atomic on POSIX and Windows (NTFS)


def train_ocr_model(
    data_dir: str,
    save_path: Path,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_weighted_sampler: bool = None,
    use_beam_val_decode: bool = None,
    use_stone_augment: bool = None,
    use_adaptive_binarize: bool = None,
    weighted_sampler_fn=None,
    max_train_batches: int = None,
    max_val_batches: int = None,
    device: str = None,
    encoder_checkpoint: Path | None = None,
    initial_model: Path | None = None,
    freeze_encoder_epochs: int = 3,
    encoder_lr_scale: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Train the Ge'ez OCR model using CTC loss.

    Args:
        data_dir:      Path to data/geez_characters/
        save_path:     Where to save best model weights
        num_epochs:    Maximum epochs (early stopping may stop sooner)
        batch_size:    Reduce to 16 if memory errors occur
        learning_rate: Initial LR (0.001 is a good starting point)
        use_weighted_sampler: Fix 1 — WeightedRandomSampler (config default)
        use_beam_val_decode: Fix 2 — beam search for val CharAcc metric
        use_stone_augment: Fix 3 — albumentations stone domain augment
        use_adaptive_binarize: Fix 4 — stored on checkpoint metadata only
        weighted_sampler_fn: Callable(dataset) → sampler (from train_ocr.py)
        max_train_batches: Optional cap for ablation speed (None = full epoch)
        max_val_batches: Optional cap for ablation speed (None = full val)
        device: Optional explicit override ("cpu" or "cuda"), e.g. from
            train_ocr.py's --device CLI flag. If provided, takes precedence
            over COMPUTE_TIER. If None (the default, e.g. when called from
            somewhere that doesn't pass it), falls back to resolving
            COMPUTE_TIER from config — same tier-based pattern used by the
            classifier/YOLO/restoration paths.

    Returns:
        dict with epoch_metrics list and best_val_loss
    """
    if use_weighted_sampler is None:
        use_weighted_sampler = OCR_USE_WEIGHTED_SAMPLER
    if use_beam_val_decode is None:
        use_beam_val_decode = OCR_USE_BEAM_DECODE
    if use_stone_augment is None:
        use_stone_augment = OCR_USE_STONE_AUGMENT
    if use_adaptive_binarize is None:
        use_adaptive_binarize = OCR_USE_ADAPTIVE_BINARIZE

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is not None:
        # Explicit override requested (e.g. train_ocr.py's --device flag).
        # Same safe-fallback philosophy as resolve_device(): never silently
        # try to run on a CUDA device that isn't actually there.
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "--device cuda requested but no CUDA device is available — "
                "falling back to CPU."
            )
            resolved_device, effective_tier = torch.device("cpu"), "cpu"
        else:
            resolved_device = torch.device(device)
            effective_tier = "gpu" if device == "cuda" else "cpu"
    else:
        resolved_device, effective_tier = resolve_device(COMPUTE_TIER)
    device = resolved_device  # unify downstream variable name regardless of which path set it

    logger.info("=" * 60)
    logger.info("STARTING GE'EZ OCR MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data:   {data_dir}")
    logger.info(f"Model:  {save_path}")
    logger.info(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
    logger.info(f"Device: {device} (effective_tier={effective_tier})")
    logger.info(
        f"Flags: weighted_sampler={use_weighted_sampler}, "
        f"beam_val={use_beam_val_decode}, stone_aug={use_stone_augment}, "
        f"adaptive_bin={use_adaptive_binarize}"
    )

    train_dataset = HHDEthiopicDataset(
        data_dir,
        split="train",
        augment=True,
        use_stone_augment=use_stone_augment,
    )
    val_dataset = HHDEthiopicDataset(
        data_dir,
        split="val",
        augment=False,
        use_stone_augment=False,
    )

    if len(train_dataset) == 0:
        logger.error("No training data found.")
        logger.error(f"Run download_hhd_ethiopic('{data_dir}') first.")
        return {"epoch_metrics": [], "best_val_loss": None}

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    if use_weighted_sampler:
        if weighted_sampler_fn is None:
            from scripts.train_ocr import create_weighted_sampler

            weighted_sampler_fn = create_weighted_sampler
        sampler = weighted_sampler_fn(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=ctc_collate_fn,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=ctc_collate_fn,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────
    if encoder_checkpoint is not None and initial_model is not None:
        raise ValueError("encoder_checkpoint and initial_model are mutually exclusive")
    if initial_model is not None:
        model = load_ocr_model(initial_model).to(device)
        model.train()
        logger.info(f"Warm-started full OCR model: {initial_model}")
    else:
        model = GeezOCRModel(num_classes=len(GEEZ_CHARSET)).to(device)
    transferred_encoder = False
    if encoder_checkpoint is not None:
        checkpoint = torch.load(encoder_checkpoint, map_location="cpu")
        checkpoint_size = checkpoint.get("image_size")
        if checkpoint_size and list(checkpoint_size) != list(OCR_IMG_SIZE):
            raise ValueError(
                f"Encoder checkpoint image size {checkpoint_size} does not match {OCR_IMG_SIZE}"
            )
        model.cnn.load_state_dict(checkpoint["cnn_state_dict"], strict=True)
        transferred_encoder = True
        logger.info(f"Loaded isolated-glyph encoder: {encoder_checkpoint}")
        if freeze_encoder_epochs > 0:
            for parameter in model.cnn.parameters():
                parameter.requires_grad = False

    # ── CTC Loss ──────────────────────────────────────────────
    # blank=BLANK_IDX: which class index represents the blank token
    # reduction='mean': average loss over batch (not sum)
    # zero_infinity=True: handle numerical instability gracefully
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    # ── Optimizer ─────────────────────────────────────────────
    if transferred_encoder:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.cnn.parameters(), "lr": learning_rate * encoder_lr_scale},
                {
                    "params": [
                        parameter
                        for name, parameter in model.named_parameters()
                        if not name.startswith("cnn.")
                    ],
                    "lr": learning_rate,
                },
            ],
            weight_decay=1e-4,
        )
        max_learning_rate: float | list[float] = [
            learning_rate * 10 * encoder_lr_scale,
            learning_rate * 10,
        ]
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        max_learning_rate = learning_rate * 10

    # ── LR Scheduler ──────────────────────────────────────────
    # IMPORTANT: built with the FULL original num_epochs regardless of
    # whether this run is resuming partway through. OneCycleLR computes a
    # fixed total_steps = epochs * steps_per_epoch ONCE at construction —
    # if this were instead built using only the epochs remaining after a
    # resume, the learning-rate curve would desync from where it actually
    # was and silently produce a wrong schedule. The correct way to resume
    # a OneCycleLR schedule is: construct it identically to a fresh run,
    # then restore its internal step counter via load_state_dict() below.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        epochs=num_epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,  # 10% of training = warmup phase
        anneal_strategy="cos",  # cosine annealing after warmup
    )

    # ── Training ──────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10
    epoch_metrics = []
    start_epoch = 0

    # ── Resume from checkpoint if one exists ───────────────────
    # WHAT: full-state resume (model + optimizer + scheduler + counters),
    # not just reloading model weights — reloading weights alone would NOT
    # reproduce the same training trajectory, since Adam's per-parameter
    # moment estimates and OneCycleLR's step position both have real state.
    # WHY this matters for Kaggle specifically: a killed kernel loses
    # everything in memory: this makes an interrupted run resumable at
    # exactly where it left off instead of losing all progress since the
    # last full run.
    checkpoint_path = save_path.parent / f"{save_path.stem}_resume_checkpoint.pth"
    if checkpoint_path.exists():
        logger.info(f"Found resume checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        expected_fingerprint = charset_fingerprint(GEEZ_CHARSET)
        if checkpoint.get("charset_fingerprint") != expected_fingerprint:
            raise ValueError("Resume checkpoint charset is missing or incompatible")
        if checkpoint.get("image_size") != list(OCR_IMG_SIZE):
            raise ValueError("Resume checkpoint OCR image size is incompatible")
        if checkpoint.get("ctc_seq_len") != OCR_CTC_SEQ_LEN:
            raise ValueError("Resume checkpoint CTC sequence length is incompatible")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        epoch_metrics = checkpoint["epoch_metrics"]
        logger.info(
            f"Resuming from epoch {start_epoch + 1}/{num_epochs} — "
            f"best_val_loss={best_val_loss:.4f}, patience_counter={patience_counter}"
        )
    else:
        logger.info("No resume checkpoint found — starting from epoch 1")

    print(f"\nTraining on {len(train_dataset)} images...")
    print(
        f"Estimated time per epoch: "
        f"~{len(train_loader) * batch_size // 60} minutes on {effective_tier.upper()}"
    )
    print("Press Ctrl+C to stop (best model is saved automatically)\n")

    for epoch in range(start_epoch, num_epochs):
        if transferred_encoder and epoch == freeze_encoder_epochs:
            for parameter in model.cnn.parameters():
                parameter.requires_grad = True
            logger.info(f"Unfroze transferred visual encoder at epoch {epoch + 1}")
        # ── Train ─────────────────────────────────────────────
        model.train()
        train_losses = []

        for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(
            train_loader
        ):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            optimizer.zero_grad()

            # Only images move to device — labels/input_lengths/label_lengths
            # stay on CPU. This is the correct, documented pattern for
            # nn.CTCLoss, not a workaround: its length tensors are expected
            # on CPU regardless of what device log_probs/targets are on.
            # Moving them would risk the exact device-mismatch gotcha CTC
            # is known for, not prevent it.
            images = images.to(device)

            # Forward pass
            # log_probs shape: (seq_len, batch, num_classes)
            log_probs = model(images)

            # CTC loss
            # Note: log_probs must be (T, N, C) — time, batch, classes
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            # Handle NaN/Inf (can occur with CTC on very short sequences)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping batch {batch_idx}: loss={loss.item():.4f}")
                continue

            loss.backward()

            # Gradient clipping: prevents exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            # Progress bar
            if (batch_idx + 1) % 20 == 0:
                avg = np.mean(train_losses[-20:])
                print(
                    f"\r  Epoch {epoch + 1:3d} "
                    f"[{batch_idx + 1:4d}/{len(train_loader)}] "
                    f"Loss: {avg:.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}",
                    end="",
                    flush=True,
                )

        avg_train_loss = np.mean(train_losses) if train_losses else 0

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_losses = []
        val_references: list[str] = []
        val_predictions: list[str] = []

        with torch.no_grad():
            for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(
                val_loader
            ):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break
                images = images.to(device)
                log_probs = model(images)
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())

                # Decode for val metric — beam when configured, else greedy
                if use_beam_val_decode:
                    texts = model.decode_beam(log_probs)
                else:
                    texts = model.decode_greedy(log_probs)

                # Compare with ground truth
                label_offset = 0
                for b, text in enumerate(texts):
                    ll = label_lengths[b].item()
                    true_l = labels[label_offset : label_offset + ll].tolist()
                    true_t = "".join(IDX_TO_CHAR.get(i, "") for i in true_l)
                    label_offset += ll

                    val_references.append(true_t)
                    val_predictions.append(text)

        avg_val_loss = np.mean(val_losses) if val_losses else 0
        sequence_metrics = compute_sequence_metrics(val_references, val_predictions)
        char_accuracy = sequence_metrics.character_accuracy

        print(
            f"\nEpoch {epoch + 1:3d}/{num_epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"CER: {sequence_metrics.cer:.1%} | "
            f"CharAcc: {char_accuracy:.1%} | "
            f"SeqAcc: {sequence_metrics.sequence_accuracy:.1%}"
        )

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "cer": float(sequence_metrics.cer),
                "char_accuracy": float(char_accuracy),
                "sequence_accuracy": float(sequence_metrics.sequence_accuracy),
            }
        )

        # ── Resume checkpoint (every epoch, atomic) ─────────────
        # Separate from the best-weights save below — this exists purely
        # so an interrupted session (Kaggle kernel dying, Wi-Fi dropping
        # mid-run) costs at most one epoch's worth of progress, not the
        # whole run. Saved every epoch regardless of whether it improved,
        # since "where training actually stopped" and "the best model so
        # far" are genuinely different things and both need to be
        # recoverable.
        _save_resume_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            epoch_metrics=epoch_metrics,
        )

        # ── Checkpoint ────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_ocr_model(model, save_path)
            print(
                f"  Saved (val_loss={avg_val_loss:.4f}, char_acc={char_accuracy:.1%})"
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {save_path}")

    # Training finished normally (full epochs or early-stopped) — the
    # resume checkpoint's job is done. Remove it so a LATER, unrelated
    # training run using the same save_path doesn't silently resume from
    # this finished run's final state instead of starting fresh.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Removed resume checkpoint (training finished): {checkpoint_path}")

    return {
        "epoch_metrics": epoch_metrics,
        "best_val_loss": best_val_loss,
        "flags": {
            "use_weighted_sampler": use_weighted_sampler,
            "use_beam_val_decode": use_beam_val_decode,
            "use_stone_augment": use_stone_augment,
            "use_adaptive_binarize": use_adaptive_binarize,
            "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
            "initial_model": str(initial_model) if initial_model else None,
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "encoder_lr_scale": encoder_lr_scale,
            "seed": seed,
        },
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — INSCRIPTION DATABASE & TRANSLATION
# ═══════════════════════════════════════════════════════════════


class InscriptionDatabase:
    """
    Database of known Ge'ez inscriptions and word translations.

    Two functions:
    1. Cross-reference: Does this recognized text match a known inscription?
    2. Translation lookup: What does this word/phrase mean?

    The database is a JSON file with two sections:
    {
        "inscriptions": [
            {
                "id": "LAL_001",
                "text": "ሰላም ለኪ...",
                "translation_en": "Hail to you...",
                "translation_am": "ሰላምታ ለሽ...",
                "site": "Lalibela, Beta Maryam",
                "date": "12th century CE",
                "significance": "Dedicatory formula, common in..."
            }
        ],
        "lexicon": {
            "ሰላም": {"en": "peace/greeting", "am": "ሰላም", "pos": "noun"},
            "ኢትዮጵያ": {"en": "Ethiopia", "am": "ኢትዮጵያ", "pos": "proper noun"}
        }
    }
    """

    def __init__(self, db_path: Path = None):
        self.inscriptions = []
        self.lexicon = {}

        if db_path and Path(db_path).exists():
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.inscriptions = data.get("inscriptions", [])
                self.lexicon = data.get("lexicon", {})
            logger.info(
                f"Inscription DB: {len(self.inscriptions)} inscriptions, "
                f"{len(self.lexicon)} lexicon entries"
            )
        else:
            logger.warning("No inscription database found — creating sample")
            self._create_sample_db()

    def _create_sample_db(self):
        """Minimal sample database for testing."""
        self.inscriptions = [
            {
                "id": "ETH_001",
                "text": "ሰላም",
                "translation_en": "Peace / Greeting",
                "translation_am": "ሰላም",
                "site": "General Ethiopic inscriptions",
                "date": "Various periods",
                "significance": "Most common dedicatory word in Ethiopian "
                "Christian inscriptions",
            },
            {
                "id": "AXU_001",
                "text": "ዓጼ",
                "translation_en": "Emperor / King of Kings",
                "translation_am": "ዓጼ",
                "site": "Aksumite royal inscriptions",
                "date": "1st–7th century CE",
                "significance": "Royal title used in Aksumite inscriptions",
            },
            {
                "id": "LAL_001",
                "text": "ማርያም",
                "translation_en": "Mary (Virgin Mary)",
                "translation_am": "ማርያም",
                "site": "Lalibela rock-hewn churches",
                "date": "12th century CE",
                "significance": "Dedicatory inscription to the Virgin Mary, "
                "common in Zagwe dynasty churches",
            },
        ]

        self.lexicon = {
            "ሰላም": {"en": "peace, greeting", "am": "ሰላምታ", "pos": "noun"},
            "ዓጼ": {"en": "emperor, king", "am": "ንጉሥ", "pos": "noun"},
            "ማርያም": {"en": "Mary", "am": "ማርያም", "pos": "proper noun"},
            "ኢትዮጵያ": {"en": "Ethiopia", "am": "ኢትዮጵያ", "pos": "proper noun"},
            "ክርስቶስ": {"en": "Christ", "am": "ክርስቶስ", "pos": "proper noun"},
            "አምላክ": {"en": "God", "am": "አምላክ", "pos": "noun"},
            "ቅዱስ": {"en": "holy, saint", "am": "ቅዱስ", "pos": "adjective"},
        }

    def translate(self, text: str) -> dict:
        """
        Translate recognized Ge'ez text.

        Looks up word by word. Returns translation for known words,
        marks unknown words explicitly.

        Args:
            text: Recognized Ge'ez character sequence

        Returns:
            dict:
                'translation_en': English translation
                'translation_am': Amharic translation/gloss
                'word_results':   List of per-word translation dicts
                'coverage':       Fraction of words that were translated
        """
        words = text.split() if " " in text else list(text)
        word_results = []
        translated = 0

        for word in words:
            if word in self.lexicon:
                entry = self.lexicon[word]
                word_results.append(
                    {
                        "word": word,
                        "en": entry.get("en", ""),
                        "am": entry.get("am", ""),
                        "pos": entry.get("pos", ""),
                        "known": True,
                    }
                )
                translated += 1
            else:
                word_results.append(
                    {"word": word, "en": "[unknown]", "am": "[unknown]", "known": False}
                )

        coverage = translated / len(words) if words else 0
        translation_en = (
            " ".join(r["en"] for r in word_results if r["known"])
            or "[unrecognized inscription]"
        )

        return {
            "translation_en": translation_en,
            "word_results": word_results,
            "coverage": coverage,
        }

    def find_inscription_match(self, text: str, threshold: float = 0.65) -> dict | None:
        """
        Check if recognized text matches a known inscription.

        Uses character-level sequence matching to handle partial
        recognition (the OCR may miss some characters).

        Args:
            text: Recognized text from OCR
            threshold: Minimum similarity ratio (0–1)

        Returns:
            Best matching inscription dict, or None if no match
        """
        from difflib import SequenceMatcher

        best_match = None
        best_ratio = 0.0

        for inscription in self.inscriptions:
            ref_text = inscription.get("text", "")
            ratio = SequenceMatcher(None, text, ref_text).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = inscription

        if best_ratio >= threshold:
            return {"match": best_match, "similarity": best_ratio, "is_new": False}

        # No match found — this is potentially a NEW undocumented inscription
        return {
            "match": None,
            "similarity": best_ratio,
            "is_new": True,
            "note": "No database match — flagged as potential new discovery",
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — FULL OCR INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════


class GeezOCRPipeline:
    """
    Complete OCR pipeline: takes a camera frame, returns all
    recognized Ge'ez text with translations and database matches.

    Usage:
        pipeline = GeezOCRPipeline()
        pipeline.load_model()

        result = pipeline.process_image(frame)
        for region in result['regions']:
            print(region['text'], region['translation']['translation_en'])
    """

    def __init__(self):
        self.model: GeezOCRModel | None = None
        self.db = InscriptionDatabase(INSCRIPTIONS_JSON)
        self.model_loaded = False

    def load_model(self, model_path: Path = None):
        """
        Load the trained OCR model.

        Args:
            model_path: Path to .pth file. Defaults to OCR_MODEL_PATH from config.
        """
        path = model_path or OCR_MODEL_PATH

        if not Path(path).exists():
            logger.warning(f"OCR model not found at {path}")
            logger.warning("Using untrained model — results will be random")
            logger.warning("Train first with: train_ocr_model(data_dir, save_path)")
            self.model = GeezOCRModel()
        else:
            self.model = load_ocr_model(path)

        self.model.eval()
        self.model_loaded = True
        logger.info("OCR pipeline ready")

    def process_image(self, image: np.ndarray) -> dict:
        """
        Full pipeline: detect text regions → OCR → translate → cross-reference.

        Args:
            image: BGR camera frame or artefact scan image

        Returns:
            result dict:
                'regions': List of dicts per detected text region:
                    {
                        'bbox':         (x, y, w, h),
                        'text':         str,
                        'confidence':   float,
                        'reliable':     bool,
                        'translation':  dict (from InscriptionDatabase.translate),
                        'db_match':     dict (from find_inscription_match),
                        'is_discovery': bool
                    }
                'total_regions':      int
                'reliable_regions':   int
                'has_new_discovery':  bool
                'timestamp':          str
        """
        if not self.model_loaded:
            self.load_model()
        assert self.model is not None

        # Step 1: Find text regions in the image
        regions = detect_text_regions(image)

        results = []
        has_new_discovery = False

        for region in regions:
            crop = region["crop"]

            # Step 2: OCR — recognize characters in this region
            ocr_result = self.model.predict(crop)

            text = ocr_result["text"]
            confidence = ocr_result["confidence"]
            reliable = ocr_result["reliable"]

            # Step 3: Translate recognized text
            translation = (
                self.db.translate(text)
                if text
                else {"translation_en": "", "word_results": [], "coverage": 0}
            )

            # Step 4: Cross-reference against inscription database
            db_match = self.db.find_inscription_match(text) if text else None
            is_discovery = (
                db_match is not None and db_match.get("is_new", False) and reliable
            )

            if is_discovery:
                has_new_discovery = True
                logger.info(
                    f"POTENTIAL NEW DISCOVERY: '{text}' — no database match found"
                )

            results.append(
                {
                    "bbox": region["bbox"],
                    "text": text,
                    "confidence": confidence,
                    "reliable": reliable,
                    "translation": translation,
                    "db_match": db_match,
                    "is_discovery": is_discovery,
                    "region_conf": region["confidence"],
                }
            )

        # Sort by confidence descending
        results.sort(key=lambda r: r["confidence"], reverse=True)

        return {
            "regions": results,
            "total_regions": len(results),
            "reliable_regions": sum(1 for r in results if r["reliable"]),
            "has_new_discovery": has_new_discovery,
            "timestamp": datetime.now().isoformat(),
        }

    def process_frame_live(self, frame: np.ndarray) -> list:
        """
        Lightweight version for live video — skips database lookup
        for speed. Returns only text and confidence.

        Args:
            frame: BGR camera frame

        Returns:
            List of (bbox, text, confidence) tuples
        """
        if not self.model_loaded:
            self.load_model()
        assert self.model is not None

        regions = detect_text_regions(frame)
        results = []

        for region in regions:
            ocr_result = self.model.predict(region["crop"])
            if ocr_result["text"]:
                results.append(
                    (region["bbox"], ocr_result["text"], ocr_result["confidence"])
                )

        return results