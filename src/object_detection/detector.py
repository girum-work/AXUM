'''AXUM ROVER — Object Detection & Classification Module
======================================================
Two-stage pipeline for artefacts sitting in the storage tray:

  Stage 1 — POSITION DETECTION (OpenCV contours, no ML)
      Finds WHERE objects are in the tray frame.
      Output: bounding boxes + center coordinates.
      Fast (~1ms), runs every frame.

  Stage 2 — TYPE IDENTIFICATION (YOLOv8 nano)
      Identifies WHAT each object is from its cropped region.
      Output: class name + confidence score.
      Slower (~50ms per crop), runs once per object.

Why two stages instead of one?
    YOLOv8 alone could do both — but on a plain tray background
    with well-separated objects, contour detection gives more
    precise pixel coordinates for arm navigation than YOLO's
    bounding boxes (which have a few pixels of slack). The arm
    needs precision; the classifier needs category. Two tools,
    two jobs.

Author: Axum Rover Team'''

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
from PIL import Image
from loguru import logger
import json
import sys
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    OBJ_MODEL_PATH, OBJ_CLASSES,
    OBJ_CONFIDENCE_MIN, ARTEFACTS_JSON,
    YOLO_MODEL_PATH
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_for_inference(image: np.ndarray) -> torch.Tensor:
    """
    Convert OpenCV BGR image to normalized PyTorch tensor.

    Args:
        image: BGR numpy array from cv2 or camera

    Returns:
        tensor: Shape (1, 3, 224, 224)
    """
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor  = inference_transform(pil_img)
    return tensor.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — MOBILENETV2 CLASSIFIER (fallback / secondary)
# ═══════════════════════════════════════════════════════════════

class ArtefactClassifier(nn.Module):
    """
    MobileNetV2-based artefact classifier.
    Used as fallback when YOLOv8 confidence is below threshold,
    or when the YOLOv8 model has not been fine-tuned yet.

    Transfer learning:
        - Lower MobileNetV2 layers: frozen (keep ImageNet weights)
        - Last 3 feature blocks: fine-tuned on artefact images
        - Classification head: replaced for our 5 classes

    Input:  224×224 RGB image
    Output: class probabilities for OBJ_CLASSES
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ArtefactClassifier, self).__init__()

        if pretrained:
            self.backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
            logger.info("MobileNetV2: loaded ImageNet weights")
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 3 feature blocks for fine-tuning
        for i in range(15, 18):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes
        logger.info(f"ArtefactClassifier ready: {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, image: np.ndarray) -> dict:
        """
        Classify a single image.

        Args:
            image: BGR numpy array (any size)

        Returns:
            dict:
                'class_name':  str
                'confidence':  float (0-1)
                'class_idx':   int
                'all_probs':   dict of all class probabilities
                'reliable':    bool
        """
        self.eval()
        tensor = preprocess_for_inference(image)

        with torch.no_grad():
            logits = self.forward(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        class_idx  = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())
        class_name = OBJ_CLASSES[class_idx] \
                     if class_idx < len(OBJ_CLASSES) else f"class_{class_idx}"

        all_probs = {
            OBJ_CLASSES[i]: float(probs[i].item())
            for i in range(len(OBJ_CLASSES))
        }

        return {
            'class_name': class_name,
            'confidence': confidence,
            'class_idx':  class_idx,
            'all_probs':  all_probs,
            'reliable':   confidence >= OBJ_CONFIDENCE_MIN,
            'source':     'mobilenetv2'
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — MOBILENETV2 PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_mobilenet_model(model: ArtefactClassifier, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes':      model.num_classes,
        'class_names':      OBJ_CLASSES,
    }, path)
    logger.info(f"MobileNetV2 model saved: {path}")


def load_mobilenet_model(path: Path) -> ArtefactClassifier:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    checkpoint  = torch.load(path, map_location='cpu')
    num_classes = checkpoint.get('num_classes', len(OBJ_CLASSES))
    model = ArtefactClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"MobileNetV2 model loaded: {path}")
    return model


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — YOLOV8 CLASSIFIER (primary)
# ═══════════════════════════════════════════════════════════════

class YOLOArtefactClassifier:
    """
    YOLOv8 nano-based artefact type identification.

    Primary classifier — runs on cropped object regions from the tray.
    Falls back to MobileNetV2 if confidence is below threshold.

    Modes:
        fine-tuned: trained on your artefact images (best accuracy)
        coco:       pretrained YOLOv8 nano with class mapping (no training needed)
    """

    COCO_TO_ARTEFACT = {
        'vase':         'pottery',
        'bowl':         'pottery',
        'cup':          'pottery',
        'bottle':       'pottery',
        'book':         'inscription_fragment',
        'clock':        'coin',
        'frisbee':      'coin',
        'sports ball':  'coin',
    }

    def __init__(
        self,
        model_path: Path = None,
        fallback_model: ArtefactClassifier = None
    ):
        """
        Args:
            model_path:     Path to fine-tuned .pt file (None = use COCO)
            fallback_model: ArtefactClassifier to use when YOLO confidence
                            is below OBJ_CONFIDENCE_MIN
        """
        self.model         = None
        self.fine_tuned    = False
        self.model_path    = model_path or YOLO_MODEL_PATH
        self.fallback      = fallback_model

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            if self.model_path and Path(self.model_path).exists():
                self.model      = YOLO(str(self.model_path))
                self.fine_tuned = True
                logger.info(f"YOLOv8 fine-tuned loaded: {self.model_path}")
            else:
                self.model      = YOLO('yolov8n.pt')
                self.fine_tuned = False
                logger.info("YOLOv8 nano (COCO fallback) loaded")

        except ImportError:
            logger.warning("ultralytics not installed — YOLOv8 unavailable")
            logger.warning("Run: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"YOLOv8 load failed: {e}")
            self.model = None

    def classify_crop(self, crop: np.ndarray) -> dict:
        """
        Classify a cropped image of one tray object.

        If YOLO result is unreliable AND a fallback MobileNetV2
        model is provided, the fallback result is returned instead.

        Args:
            crop: BGR image of a single object

        Returns:
            dict:
                'class_name':  str
                'confidence':  float
                'reliable':    bool
                'source':      str
        """
        if self.model is None:
            return self._try_fallback(crop) or self._unknown_result()

        try:
            results = self.model(crop, verbose=False, conf=0.25)

            if not results or len(results) == 0:
                return self._try_fallback(crop) or self._unknown_result()

            result = results[0]

            if self.fine_tuned:
                parsed = self._parse_finetuned(result)
            else:
                parsed = self._parse_coco(result)

            # If not reliable, try MobileNetV2 fallback
            if not parsed['reliable'] and self.fallback:
                fallback_result = self.fallback.predict(crop)
                if fallback_result['confidence'] > parsed['confidence']:
                    return fallback_result

            return parsed

        except Exception as e:
            logger.warning(f"YOLOv8 inference error: {e}")
            return self._try_fallback(crop) or self._unknown_result()

    def _parse_finetuned(self, result) -> dict:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._unknown_result()

        confidences = boxes.conf.cpu().numpy()
        best_idx    = confidences.argmax()
        confidence  = float(confidences[best_idx])
        class_idx   = int(boxes.cls[best_idx].item())
        class_name  = result.names.get(class_idx, 'other')

        return {
            'class_name': class_name,
            'confidence': confidence,
            'reliable':   confidence >= OBJ_CONFIDENCE_MIN,
            'source':     'yolo_finetuned'
        }

    def _parse_coco(self, result) -> dict:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._unknown_result()

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].item())
            class_idx  = int(boxes.cls[i].item())
            class_name = result.names.get(class_idx, '').lower()

            if class_name in self.COCO_TO_ARTEFACT:
                return {
                    'class_name': self.COCO_TO_ARTEFACT[class_name],
                    'confidence': confidence * 0.7,
                    'reliable':   confidence >= 0.70,
                    'source':     'yolo_coco'
                }

        return {
            'class_name': 'other',
            'confidence': 0.40,
            'reliable':   False,
            'source':     'yolo_coco'
        }

    def _try_fallback(self, crop: np.ndarray) -> dict | None:
        if self.fallback and crop is not None:
            try:
                return self.fallback.predict(crop)
            except Exception:
                pass
        return None

    def _unknown_result(self) -> dict:
        return {
            'class_name': 'unknown',
            'confidence': 0.0,
            'reliable':   False,
            'source':     'fallback'
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — TRAY POSITION DETECTION (OpenCV contours)
# ═══════════════════════════════════════════════════════════════

def detect_objects_in_tray(frame: np.ndarray) -> list:
    """
    Find pixel positions of objects in the storage tray.

    Uses adaptive thresholding + contour detection.
    Assumes a plain light-coloured tray background.
    Objects appear darker than the tray surface.

    Args:
        frame: BGR camera frame looking down at the tray

    Returns:
        List of dicts, sorted left-to-right:
            {
                'bbox':    (x, y, w, h),
                'center':  (cx, cy),
                'area':    float,
                'contour': np.ndarray
            }
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold: handles uneven lighting across the tray
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=10
    )

    # Remove small noise
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects    = []
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by size: 0.5% to 30% of frame
        if not (frame_area * 0.005 < area < frame_area * 0.30):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        objects.append({
            'bbox':    (x, y, w, h),
            'center':  (cx, cy),
            'area':    area,
            'contour': contour
        })

    # Sort left to right (processing order for arm)
    objects.sort(key=lambda o: o['center'][0])
    logger.info(f"Tray: {len(objects)} objects detected")
    return objects


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — COMBINED PIPELINE
# ═══════════════════════════════════════════════════════════════

def detect_and_classify_tray(
    frame: np.ndarray,
    classifier: YOLOArtefactClassifier
) -> list:
    """
    Complete tray analysis: positions + type identification.

    Stage 1: detect_objects_in_tray() — finds positions (fast, always runs)
    Stage 2: classifier.classify_crop() — identifies type (per object)

    Args:
        frame:      BGR camera frame of the tray from above
        classifier: Loaded YOLOArtefactClassifier instance

    Returns:
        List of dicts, one per detected object:
        {
            'bbox':       (x, y, w, h),
            'center':     (cx, cy),
            'area':       float,
            'class_name': str,
            'confidence': float,
            'reliable':   bool,
            'source':     str
        }
    """
    objects = detect_objects_in_tray(frame)
    if not objects:
        return []

    results = []
    for obj in objects:
        x, y, w, h = obj['bbox']
        crop        = frame[y:y+h, x:x+w]

        if crop.shape[0] < 20 or crop.shape[1] < 20:
            cls = {'class_name': 'unknown', 'confidence': 0.0,
                   'reliable': False, 'source': 'too_small'}
        else:
            cls = classifier.classify_crop(crop)

        results.append({
            'bbox':       obj['bbox'],
            'center':     obj['center'],
            'area':       obj['area'],
            'class_name': cls['class_name'],
            'confidence': cls['confidence'],
            'reliable':   cls['reliable'],
            'source':     cls['source']
        })

    logger.info(
        "Tray classified: "
        + ", ".join(f"{r['class_name']}({r['confidence']:.0%})"
                    for r in results)
    )
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — VISUAL OVERLAY
# ═══════════════════════════════════════════════════════════════

def draw_detection_overlay(
    frame: np.ndarray,
    objects: list,
    classifications: list = None
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the tray view.

    Args:
        frame:           Original BGR frame
        objects:         From detect_objects_in_tray()
        classifications: Optional results from detect_and_classify_tray()

    Returns:
        Annotated BGR frame
    """
    overlay = frame.copy()

    for i, obj in enumerate(objects):
        x, y, w, h = obj['bbox']
        cx, cy      = obj['center']

        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(overlay, f"#{i+1}", (x+4, y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if classifications and i < len(classifications):
            cls   = classifications[i]
            label = f"{cls.get('class_name','?')} ({cls.get('confidence',0):.0%})"
            color = (0, 255, 0) if cls.get('reliable') else (0, 165, 255)
            cv2.putText(overlay, label, (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.putText(overlay, f"Objects: {len(objects)}",
                (8, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — ARTEFACT DATABASE
# ═══════════════════════════════════════════════════════════════

class ArtefactDatabase:
    """
    Cross-references detected artefacts against a JSON database
    of known Ethiopian cultural objects.

    Database format (data/artefact_database.json):
    [
        {
            "id": "ETH_POT_001",
            "type": "pottery",
            "culture": "Aksumite",
            "period": "4th-7th century CE",
            "description": "...",
            "significance": "high",
            "keywords": ["pottery", "aksumite", "storage"]
        },
        ...
    ]
    """

    def __init__(self, db_path: Path = None):
        self.entries = []
        path = db_path or ARTEFACTS_JSON

        if path and Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                self.entries = json.load(f)
            logger.info(f"Artefact DB loaded: {len(self.entries)} entries")
        else:
            logger.warning("Artefact database not found — using built-in samples")
            self._load_samples()

    def _load_samples(self):
        self.entries = [
            {
                "id": "ETH_POT_001",
                "type": "pottery",
                "culture": "Aksumite",
                "period": "4th-7th century CE",
                "description": "Red-burnished storage vessel",
                "significance": "high",
                "keywords": ["pottery", "aksumite", "ceramic"]
            },
            {
                "id": "ETH_COIN_001",
                "type": "coin",
                "culture": "Aksumite Kingdom",
                "period": "3rd-7th century CE",
                "description": "Gold Aksumite coin with royal portrait",
                "significance": "critical",
                "keywords": ["coin", "aksumite", "gold", "royal"]
            },
            {
                "id": "ETH_STONE_001",
                "type": "stone_carving",
                "culture": "Pre-Aksumite",
                "period": "1st millennium BCE",
                "description": "Stone tablet with inscription",
                "significance": "critical",
                "keywords": ["stone", "inscription", "tablet"]
            }
        ]

    def find_matches(
        self,
        class_name: str,
        confidence: float,
        threshold: float = 0.70
    ) -> list:
        """Return top 3 database entries matching the detected class."""
        from difflib import SequenceMatcher
        matches = []

        for entry in self.entries:
            type_match     = entry['type'] == class_name
            keyword_scores = [
                SequenceMatcher(None, class_name, kw).ratio()
                for kw in entry.get('keywords', [])
            ]
            keyword_score = max(keyword_scores) if keyword_scores else 0.0

            relevance = (0.6 + 0.4 * confidence) if type_match \
                        else (keyword_score * confidence)

            if relevance >= threshold:
                matches.append({
                    'entry':      entry,
                    'relevance':  relevance,
                    'type_match': type_match
                })

        matches.sort(key=lambda m: m['relevance'], reverse=True)
        return matches[:3]


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MOBILENETV2 TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

class ArtefactDataset(torch.utils.data.Dataset):
    """
    Dataset loader for artefact classification training.

    Expected folder structure:
        data/artefact_classes/
            pottery/
                img_001.jpg ...
            stone_carving/
                img_001.jpg ...
            coin/        ...
            inscription_fragment/ ...
            other/       ...
    """

    def __init__(self, root_dir: str, transform=None, split: str = 'train'):
        self.root_dir  = Path(root_dir)
        self.transform = transform or (
            train_transform if split == 'train' else inference_transform
        )
        self.samples   = []

        for class_idx, class_name in enumerate(OBJ_CLASSES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory missing: {class_dir}")
                continue

            images    = (list(class_dir.glob("*.jpg")) +
                         list(class_dir.glob("*.jpeg")) +
                         list(class_dir.glob("*.png")))
            split_idx = int(len(images) * 0.85)
            images    = images[:split_idx] if split == 'train' \
                        else images[split_idx:]

            for img_path in images:
                self.samples.append((img_path, class_idx))

        logger.info(f"ArtefactDataset ({split}): {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx


def train_artefact_classifier(
    data_dir:      str,
    save_path:     Path,
    num_epochs:    int   = 30,
    batch_size:    int   = 16,
    learning_rate: float = 0.001
):
    """
    Train the MobileNetV2 artefact classifier.

    Args:
        data_dir:      Path to data/artefact_classes/
        save_path:     Where to save the best model
        num_epochs:    Max epochs (early stopping may end sooner)
        batch_size:    Reduce to 8 if out of memory
        learning_rate: Initial learning rate
    """
    logger.info("Training ArtefactClassifier (MobileNetV2)...")

    train_dataset = ArtefactDataset(data_dir, split='train')
    val_dataset   = ArtefactDataset(data_dir, split='val',
                                    transform=inference_transform)

    if len(train_dataset) == 0:
        logger.error("No training images found. Check folder structure.")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    model     = ArtefactClassifier(num_classes=len(OBJ_CLASSES), pretrained=True)
    criterion = nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    early_stop       = 7

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = torch.max(outputs, 1)
            train_total   += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc   = val_correct / val_total if val_total > 0 else 0

        scheduler.step(avg_val)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train: {avg_train:.4f} | "
              f"Val: {avg_val:.4f} | "
              f"Val Acc: {val_acc:.1%}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            save_mobilenet_model(model, save_path)
            print(f"  ✓ Saved (val_loss={avg_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model: {save_path}")
    return model


if __name__ == "__main__":
    print("Object Detection Module")
    print(f"Classes: {OBJ_CLASSES}")
    print(f"YOLOv8 model path: {YOLO_MODEL_PATH}")
    print(f"MobileNetV2 model path: {OBJ_MODEL_PATH}")
    print("\nTo test tray detection on a webcam frame:")
    print("  cap = cv2.VideoCapture(0)")
    print("  ret, frame = cap.read()")
    print("  objects = detect_objects_in_tray(frame)")
    # ═══════════════════════════════════════════════════════════════
# SECTION 10 — YOLOV8 TYPE CLASSIFIER
# ═══════════════════════════════════════════════════════════════

class YOLOArtefactClassifier:
    """
    YOLOv8 nano-based artefact type identification.

    Runs on CROPPED regions from the tray (output of detect_objects_in_tray).
    Does NOT detect positions — contour detection handles that.

    Two modes:
      1. Pre-trained fallback: uses YOLOv8 nano's built-in object classes
         to do rough identification (e.g., "vase" maps to "pottery")
         Works immediately, no training needed, lower accuracy.

      2. Fine-tuned mode (recommended): train on your artefact images.
         Use this once you have 50+ images per class.
         Higher accuracy, domain-specific.

    Model file: models/yolov8_artefacts.pt
    If not found, falls back to MobileNetV2 (ArtefactClassifier).
    """

    # Mapping from YOLOv8 COCO class names to our artefact classes
    # Used only in fallback mode (before fine-tuning)
    COCO_TO_ARTEFACT = {
        'vase':       'pottery',
        'bowl':       'pottery',
        'cup':        'pottery',
        'bottle':     'pottery',
        'book':       'inscription_fragment',
        'clock':      'coin',
        'frisbee':    'coin',
        'sports ball':'coin',
    }

    def __init__(self, model_path: Path = None):
        """
        Args:
            model_path: Path to fine-tuned .pt file.
                        If None or not found, uses COCO fallback.
        """
        self.model        = None
        self.fine_tuned   = False
        self.model_path   = model_path

        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model — fine-tuned if available, nano otherwise."""
        if self.model_path and Path(self.model_path).exists():
            try:
                self.model      = YOLO(str(self.model_path))
                self.fine_tuned = True
                logger.info(f"YOLOv8 fine-tuned model loaded: {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Fine-tuned model load failed: {e}")

        # Fall back to YOLOv8 nano (COCO pretrained)
        try:
            self.model      = YOLO('yolov8n.pt')  # downloads ~6MB if needed
            self.fine_tuned = False
            logger.info("YOLOv8 nano (COCO) loaded — using class mapping fallback")
        except Exception as e:
            logger.error(f"YOLOv8 load failed entirely: {e}")
            self.model = None

    def classify_crop(self, crop: np.ndarray) -> dict:
        """
        Classify a cropped image of ONE object from the tray.

        Args:
            crop: BGR image of a single object (from detect_objects_in_tray)

        Returns:
            dict:
                'class_name':  str   — artefact type
                'confidence':  float — 0 to 1
                'reliable':    bool  — True if confidence >= threshold
                'source':      str   — 'yolo_finetuned' or 'yolo_coco' or 'fallback'
        """
        if self.model is None:
            return self._fallback_result()

        try:
            # Run inference on the crop
            # verbose=False suppresses per-frame console output
            results = self.model(crop, verbose=False, conf=0.25)

            if not results or len(results) == 0:
                return self._fallback_result()

            result = results[0]

            if self.fine_tuned:
                return self._parse_finetuned_result(result)
            else:
                return self._parse_coco_result(result)

        except Exception as e:
            logger.warning(f"YOLOv8 inference error: {e}")
            return self._fallback_result()

    def _parse_finetuned_result(self, result) -> dict:
        """
        Parse result from our fine-tuned model.
        Class names are our artefact categories directly.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._fallback_result()

        # Get highest-confidence detection
        confidences = boxes.conf.cpu().numpy()
        best_idx    = confidences.argmax()
        confidence  = float(confidences[best_idx])
        class_idx   = int(boxes.cls[best_idx].item())
        class_name  = result.names.get(class_idx, 'other')

        return {
            'class_name': class_name,
            'confidence': confidence,
            'reliable':   confidence >= 0.60,
            'source':     'yolo_finetuned'
        }

    def _parse_coco_result(self, result) -> dict:
        """
        Parse result from COCO-pretrained model.
        Maps COCO class names to artefact categories.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._fallback_result()

        # Check all detections for known artefact-related classes
        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].item())
            class_idx  = int(boxes.cls[i].item())
            class_name = result.names.get(class_idx, '').lower()

            if class_name in self.COCO_TO_ARTEFACT:
                return {
                    'class_name': self.COCO_TO_ARTEFACT[class_name],
                    'confidence': confidence * 0.7,  # penalty for indirect mapping
                    'reliable':   confidence >= 0.70,
                    'source':     'yolo_coco'
                }

        # No artefact class found — return 'other'
        return {
            'class_name': 'other',
            'confidence': 0.40,
            'reliable':   False,
            'source':     'yolo_coco'
        }

    def _fallback_result(self) -> dict:
        """Return when model unavailable or inference fails."""
        return {
            'class_name': 'unknown',
            'confidence': 0.0,
            'reliable':   False,
            'source':     'fallback'
        }