# src/ocr/model.py
"""
AXUM ROVER — Ge'ez OCR Model Architecture
==========================================
Recognizes Ge'ez (Ethiopic) characters from images of stone inscriptions.

The Ge'ez writing system (Fidel) is an abugida — each character represents
a consonant-vowel combination. There are 231 base characters in the standard
Ethiopic Unicode block (U+1200 to U+137F), plus punctuation and numerals.

Architecture: CNN + LSTM (Convolutional + Recurrent Neural Network)
─────────────────────────────────────────────────────────────────────
Why not just a CNN classifier?
    A pure CNN treats each character independently. But Ge'ez characters
    in inscriptions are often connected or partially overlapping — the same
    visual region might contain parts of two characters. A CNN alone
    would misclassify these boundaries.

    The LSTM layer adds sequential context: it reads the CNN features
    left-to-right and uses surrounding character context to resolve
    ambiguous individual characters. This is the same approach used
    in Google's production OCR systems.

Why not use a Transformer (like TrOCR)?
    Transformers require enormous training data (millions of samples)
    and significant compute for inference. On your CPU-only laptop,
    a Transformer-based OCR model would take 10+ seconds per image.
    The CNN+LSTM approach runs in ~0.3 seconds per image on CPU —
    fast enough for the artefact handling workflow.

Input:  Grayscale image of a text region, resized to 32×128 pixels
        (height × width — taller than wide for character strips)
Output: Sequence of character class probabilities (CTC output)

Author: Axum Rover Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NUM_GEEZ_CLASSES, OCR_BEAM_WIDTH, OCR_IMG_SIZE, OCR_USE_BEAM_DECODE
from src.ocr.training_contract import build_assigned_ethiopic_charset, charset_fingerprint


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CHARACTER SET DEFINITION
# ═══════════════════════════════════════════════════════════════

def build_geez_charset() -> tuple:
    """
    Build the complete Ge'ez character set from Unicode.

    The Ethiopic Unicode block covers U+1200 to U+137F.
    This includes:
    - Syllables (ሀ to ፗ): the main character set, 7 orders per base
    - Numerals (፩ to ፼)
    - Punctuation (፡ ። ፣ ፤ ፥ ፦)
    - Extended Ethiopic (U+2D80–U+2DDF) for some dialects

    Returns:
        charset: List of all Ge'ez characters (strings)
        char_to_idx: Dict mapping character → integer index
        idx_to_char: Dict mapping integer index → character
    """
    charset = list(build_assigned_ethiopic_charset())

    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    idx_to_char = {idx: char for idx, char in enumerate(charset)}

    logger.info(f"Ge'ez charset: {len(charset)} tokens "
                f"({len(charset)-2} characters + BLANK + UNK)")

    return charset, char_to_idx, idx_to_char


# Build charset at module import time
GEEZ_CHARSET, CHAR_TO_IDX, IDX_TO_CHAR = build_geez_charset()
BLANK_IDX = 0  # CTC blank token index


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CNN FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════

class GeezCNNFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor for Ge'ez character images.

    Takes a grayscale text strip image (1×32×128) and produces
    a sequence of feature vectors (one per column slice of the image).

    Architecture:
        Input: 1 × 32 × 128   (channels × height × width)
          ↓
        Conv1: 64 × 30 × 126   (3×3 conv, no padding)
        MaxPool: 64 × 15 × 63
          ↓
        Conv2: 128 × 13 × 61   (3×3 conv, no padding)
        MaxPool: 128 × 6 × 30
          ↓
        Conv3: 256 × 4 × 28    (3×3 conv, no padding)
        BatchNorm
          ↓
        Conv4: 256 × 4 × 28    (3×3 conv, same padding)
        BatchNorm
        MaxPool: 256 × 2 × 14  (pool height only — preserve width)
          ↓
        Conv5: 512 × 1 × 12    (3×3 conv, no padding)
        BatchNorm
          ↓
        Reshape: 512 × 12      (feature_size × sequence_length)
          ↓
        Output: sequence of 12 feature vectors, each 512-dimensional

    The width dimension (12 here) becomes the sequence length for the LSTM.
    Each of the 12 positions represents a vertical slice of the image,
    and the LSTM reads these left-to-right like reading text.
    """

    def __init__(self):
        super(GeezCNNFeatureExtractor, self).__init__()

        # Block 1: Basic edge and texture detection
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Higher-level pattern detection
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Complex feature detection with batch normalization
        # BatchNorm normalizes activations within each batch —
        # stabilizes training and allows higher learning rates
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Block 4: Same-size convolution (padding=1 keeps dimensions)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Pool only in height dimension to preserve sequence length
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # Block 5: Final feature compression (kernel height=2 collapses 2→1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(2, 3), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, 1, 32, 128)

        Returns:
            features: Shape (sequence_len, batch, 512)
                      Ready for LSTM input
        """
        x = self.conv1(x)   # → (batch, 64, 15, 63)
        x = self.conv2(x)   # → (batch, 128, 6, 30)
        x = self.conv3(x)   # → (batch, 256, 4, 28)
        x = self.conv4(x)   # → (batch, 256, 2, 28)
        x = self.conv5(x)   # → (batch, 512, 1, 26)

        # Squeeze height dimension (now = 1 after all pooling)
        # Shape: (batch, 512, 1, 26) → (batch, 512, 26)
        batch_size = x.size(0)
        x = x.squeeze(2)

        # Transpose for LSTM: needs (sequence_len, batch, features)
        # Shape: (batch, 512, 26) → (26, batch, 512)
        x = x.permute(2, 0, 1)

        return x


class GeezGlyphClassifier(nn.Module):
    """Isolated-glyph pretraining head over the shared OCR visual encoder."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = GeezCNNFeatureExtractor()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sequence = self.cnn(images)
        pooled = sequence.mean(dim=0)
        return self.classifier(pooled)


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — FULL OCR MODEL (CNN + LSTM + CTC)
# ═══════════════════════════════════════════════════════════════

class GeezOCRModel(nn.Module):
    """
    Complete Ge'ez OCR model: CNN feature extractor + BiLSTM + CTC decoder.

    The Bidirectional LSTM reads the feature sequence in both directions:
    - Forward pass:  reads left-to-right (normal reading direction)
    - Backward pass: reads right-to-left (catches context from later chars)

    Both directions are concatenated, giving each position context from
    both preceding AND following characters — significantly improving
    recognition of ambiguous characters.

    CTC (Connectionist Temporal Classification):
        CTC is a loss function specifically designed for sequence-to-sequence
        problems where the input and output sequences have different lengths
        and are not aligned. Perfect for OCR where an image of variable
        width maps to text of variable length.

        CTC introduces the BLANK token: during decoding, consecutive
        duplicates and blank tokens are collapsed:
        ['ሰ','ሰ','<B>','ሰ','ሰ','ላ','<B>','ም'] → ['ሰ','ሳ','ም'] → "ሰላም"

    Full architecture:
        Input (batch, 1, 32, 128)
          ↓
        CNN Feature Extractor → (26, batch, 512)
          ↓
        BiLSTM layer 1: hidden=256 each direction → (26, batch, 512)
        Dropout(0.3)
          ↓
        BiLSTM layer 2: hidden=256 each direction → (26, batch, 512)
        Dropout(0.3)
          ↓
        Linear(512 → num_classes) → (26, batch, num_classes)
          ↓
        LogSoftmax → log probabilities per position per class
          ↓
        CTC Decoder → final character sequence
    """

    def __init__(self, num_classes: int = None):
        super(GeezOCRModel, self).__init__()

        if num_classes is None:
            num_classes = len(GEEZ_CHARSET)

        self.num_classes = num_classes

        # CNN feature extractor
        self.cnn = GeezCNNFeatureExtractor()

        # Bidirectional LSTM
        # input_size=512: matches CNN output feature dimension
        # hidden_size=256: each direction produces 256-dim hidden state
        # num_layers=2: stacked LSTM for more complex patterns
        # batch_first=False: sequence first (required for CTC)
        # bidirectional=True: processes both directions
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=False,
            bidirectional=True,
            dropout=0.3         # dropout between LSTM layers
        )

        # Final classification layer
        # Input: 512 (256 forward + 256 backward)
        # Output: num_classes (one logit per character)
        self.classifier = nn.Linear(512, num_classes)

        # LogSoftmax for CTC loss
        # CTC requires log probabilities, not raw logits
        self.log_softmax = nn.LogSoftmax(dim=2)

        logger.info(f"GeezOCRModel initialized: {num_classes} classes, "
                    f"{self._count_params():,} parameters")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            x: Input images, shape (batch, 1, 32, 128)

        Returns:
            log_probs: Shape (seq_len, batch, num_classes)
                       Log probabilities — input to CTCLoss
        """
        # CNN: extract spatial features
        # Output: (seq_len=26, batch, 512)
        features = self.cnn(x)

        # LSTM: add sequential context
        # lstm_out shape: (seq_len, batch, 512) — 256 × 2 directions
        lstm_out, _ = self.lstm(features)

        # Linear classifier: map to character logits
        # Output: (seq_len, batch, num_classes)
        logits = self.classifier(lstm_out)

        # Log softmax for CTC
        log_probs = self.log_softmax(logits)

        return log_probs

    def decode_greedy(self, log_probs: torch.Tensor) -> list:
        """
        Greedy CTC decoding: at each time step, take the character
        with the highest probability. Then collapse duplicates and blanks.

        This is not the most accurate decoder (beam search is better)
        but it's fast and sufficient for the demo.

        Args:
            log_probs: Shape (seq_len, batch, num_classes)

        Returns:
            texts: List of decoded strings, one per batch item
        """
        # Get argmax at each time step
        # probs shape: (seq_len, batch)
        _, max_indices = torch.max(log_probs, dim=2)

        texts = []
        batch_size = max_indices.size(1)

        for b in range(batch_size):
            sequence = max_indices[:, b].tolist()
            text = self._ctc_collapse(sequence)
            texts.append(text)

        return texts

    def decode_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 10,
    ) -> list:
        """
        CTC beam search decoding — more accurate than greedy on ambiguous glyphs.

        WHAT: Maintains multiple candidate character sequences and returns the
        highest-probability collapsed string per batch item.
        WHY: Greedy decoding fails on visually similar Ge'ez syllables; beam
        search favours plausible bigrams without retraining.

        Args:
            log_probs: Shape (seq_len, batch, num_classes) from forward()
            beam_width: Candidate sequences to keep per time step (default 10)

        Returns:
            List of decoded strings, one per batch item
        """
        import numpy as np

        seq_len, batch_size, _ = log_probs.shape
        lp = log_probs.detach().cpu().numpy()
        results = []

        for b in range(batch_size):
            beams = {(): (0.0, float("-inf"))}

            for t in range(seq_len):
                candidates = lp[t, b]
                top_chars = np.argsort(candidates)[-beam_width:]
                new_beams = {}

                for prefix, (log_pb, log_pnb) in beams.items():
                    log_psum = np.logaddexp(log_pb, log_pnb)

                    for c in top_chars:
                        p = float(candidates[c])

                        if c == BLANK_IDX:
                            nb_pb, nb_pnb = new_beams.get(prefix, (-np.inf, -np.inf))
                            new_beams[prefix] = (
                                np.logaddexp(nb_pb, log_psum + p),
                                nb_pnb,
                            )
                        elif prefix and prefix[-1] == c:
                            nb_pb, nb_pnb = new_beams.get(prefix, (-np.inf, -np.inf))
                            new_beams[prefix] = (
                                nb_pb,
                                np.logaddexp(nb_pnb, log_pnb + p),
                            )
                            ext = prefix + (c,)
                            eb_pb, eb_pnb = new_beams.get(ext, (-np.inf, -np.inf))
                            new_beams[ext] = (
                                eb_pb,
                                np.logaddexp(eb_pnb, log_pb + p),
                            )
                        else:
                            ext = prefix + (c,)
                            eb_pb, eb_pnb = new_beams.get(ext, (-np.inf, -np.inf))
                            new_beams[ext] = (
                                eb_pb,
                                np.logaddexp(eb_pnb, log_psum + p),
                            )

                scored = [
                    (pfx, np.logaddexp(pb, pnb))
                    for pfx, (pb, pnb) in new_beams.items()
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                beams = {pfx: new_beams[pfx] for pfx, _ in scored[:beam_width]}

            best_prefix = max(
                beams.keys(),
                key=lambda p: np.logaddexp(beams[p][0], beams[p][1]),
            )
            results.append(self._indices_to_text(list(best_prefix)))

        return results

    def decode_beam(
        self,
        log_probs: torch.Tensor,
        beam_width: int = None,
    ) -> list:
        """
        Alias for decode_beam_search using OCR_BEAM_WIDTH from config when unset.

        Args:
            log_probs: Shape (seq_len, batch, num_classes)
            beam_width: Override beam width (default: OCR_BEAM_WIDTH from config)

        Returns:
            List of decoded strings, one per batch item
        """
        if beam_width is None:
            beam_width = OCR_BEAM_WIDTH
        return self.decode_beam_search(log_probs, beam_width=beam_width)

    def _indices_to_text(self, indices: list) -> str:
        """Convert class index list to Ge'ez string (skip BLANK/UNK)."""
        chars = []
        for idx in indices:
            if idx == BLANK_IDX:
                continue
            if idx < len(IDX_TO_CHAR):
                char = IDX_TO_CHAR[idx]
                if char not in ("<BLANK>", "<UNK>"):
                    chars.append(char)
        return "".join(chars)

    def _ctc_collapse(self, sequence: list) -> str:
        """
        Apply CTC collapsing rules:
        1. Remove consecutive duplicate characters
        2. Remove blank tokens

        Example:
            [1, 1, 0, 1, 2, 2, 0, 3] → [1, 1, 2, 3] → [1, 2, 3]
            (0 is blank, duplicates collapsed first, then blanks removed)

        Args:
            sequence: List of integer class indices

        Returns:
            text: Decoded string
        """
        # Step 1: Remove consecutive duplicates
        collapsed = []
        prev = None
        for idx in sequence:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Step 2: Remove blank tokens and convert to characters
        chars = []
        for idx in collapsed:
            if idx == BLANK_IDX:
                continue
            if idx < len(IDX_TO_CHAR):
                char = IDX_TO_CHAR[idx]
                if char not in ('<BLANK>', '<UNK>'):
                    chars.append(char)

        return ''.join(chars)

    def predict(
        self,
        image: 'np.ndarray',
        return_confidence: bool = True
    ) -> dict:
        """
        Predict text in a single image (numpy array).

        Args:
            image: Grayscale or BGR image containing Ge'ez text
            return_confidence: Whether to compute per-character confidence

        Returns:
            dict:
                'text':       str  — recognized characters
                'confidence': float — mean confidence across positions
                'reliable':   bool — True if confidence above threshold
                'char_confs': list — per-character confidence scores
        """
        self.eval()

        # Lazy import avoids circular import (pipeline imports this module)
        from src.ocr.pipeline import preprocess_for_ocr

        tensor = preprocess_for_ocr(image)

        with torch.no_grad():
            log_probs = self.forward(tensor)

        from config import OCR_CONFIDENCE_MIN, OCR_USE_MISSING_TOKENS

        if OCR_USE_MISSING_TOKENS:
            from src.ocr.ocr_postprocess import greedy_decode_with_missing

            text, confidence = greedy_decode_with_missing(
                log_probs, IDX_TO_CHAR, BLANK_IDX
            )
        elif OCR_USE_BEAM_DECODE:
            texts = self.decode_beam_search(log_probs, beam_width=OCR_BEAM_WIDTH)
            text = texts[0] if texts else ""
            probs = torch.exp(log_probs[:, 0, :])
            confidence = float(torch.max(probs, dim=1).values.mean().item())
        else:
            texts = self.decode_greedy(log_probs)
            text = texts[0] if texts else ""
            probs = torch.exp(log_probs[:, 0, :])
            confidence = float(torch.max(probs, dim=1).values.mean().item())

        probs = torch.exp(log_probs[:, 0, :])
        max_probs = torch.max(probs, dim=1).values
        char_confs = [float(p.item()) for p in max_probs if float(p.item()) > 0.1]

        return {
            'text':       text,
            'confidence': confidence,
            'reliable':   confidence >= OCR_CONFIDENCE_MIN,
            'char_confs': char_confs,
            'char_count': len(text.replace('[MISSING]', '')),
            'has_missing': '[MISSING]' in text,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_ocr_model(model: GeezOCRModel, path: Path):
    """Save OCR model weights and charset info."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build a CPU-mapped state dict regardless of what device the model
    # trained on. This is NOT the same bug as before (model.to("cpu") used
    # to mutate the live training model in place, breaking the next epoch's
    # forward pass) — that specific bug is fixed, this call site no longer
    # does that. But this function was still saving model.state_dict()
    # directly, which means a checkpoint trained on GPU would be saved with
    # CUDA tensors — loadable, but only with an explicit map_location='cpu'
    # downstream, which the rover's CPU-only deployment can't be assumed to
    # always specify. Building the state dict as CPU tensors here makes the
    # checkpoint portable by construction, independent of training device.
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        'model_state_dict': cpu_state,
        'num_classes':      model.num_classes,
        'charset_size':     len(GEEZ_CHARSET),
        'charset':          list(GEEZ_CHARSET),
        'charset_fingerprint': charset_fingerprint(GEEZ_CHARSET),
        'image_size':       list(OCR_IMG_SIZE),
    }, path)
    logger.info(f"OCR model saved: {path}")


def load_ocr_model(path: Path) -> GeezOCRModel:
    """Load a saved OCR model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OCR model not found: {path}")

    checkpoint  = torch.load(path, map_location='cpu')
    num_classes = checkpoint.get('num_classes', len(GEEZ_CHARSET))
    saved_charset = checkpoint.get('charset')
    if saved_charset is not None and list(saved_charset) != list(GEEZ_CHARSET):
        raise ValueError(
            "OCR checkpoint charset does not match the active charset "
            f"({checkpoint.get('charset_fingerprint', 'unknown')} != "
            f"{charset_fingerprint(GEEZ_CHARSET)})"
        )
    if saved_charset is None and num_classes != len(GEEZ_CHARSET):
        raise ValueError(
            f"Legacy OCR checkpoint has {num_classes} classes but the active "
            f"charset has {len(GEEZ_CHARSET)}; retraining is required"
        )
    saved_image_size = checkpoint.get('image_size')
    if saved_image_size is not None and list(saved_image_size) != list(OCR_IMG_SIZE):
        raise ValueError(
            f"OCR checkpoint image size {saved_image_size} does not match {OCR_IMG_SIZE}"
        )

    model = GeezOCRModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"OCR model loaded from {path}")
    return model