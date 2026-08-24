"""
Character-level restoration model for Ge'ez and Amharic.

Sizing note: Aeneas uses ~25-30M parameters for Latin, but it trains on far more
text than exists in Ge'ez. Our largest clean Ge'ez corpus is ~1.15M characters,
so the default here is ~5M parameters. Scaling measurements in
scripts/run_corpus_scaling.py showed returns flattening hard (8.7x data for
+1.56 points), which means capacity is not the binding constraint and a larger
model would mostly memorise.

Representation: every fidel decomposes exactly into consonant base + vowel order
(39 x 8, zero round-trip failures over 1.9M graphemes). Predicting the two
factors separately gives 47 output units instead of 238-263 whole fidels, so
each unit sees roughly five times more training signal -- which matters most for
the rare fidels that whole-symbol models never learn.

Graphemes outside the main syllabary (labialised forms, punctuation; 2.2% of
Ge'ez, 11.3% of Amharic) keep their own base id and carry vowel = NONE.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.ocr.damage import (
    MISSING_MARKER,
    NOFILL_MARKER,
    SYLLABARY_END,
    SYLLABARY_START,
    UNKNOWN_GAP_MARKER,
    split_graphemes,
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_BASES = (PAD_TOKEN, MISSING_MARKER, UNKNOWN_GAP_MARKER, UNK_TOKEN,
                 NOFILL_MARKER)

VOWEL_NONE = 8  # index after the eight real vowel orders
NUM_VOWEL_CLASSES = 9


def decompose_fidel(grapheme: str) -> tuple[str, int]:
    """
    Split a grapheme into its base consonant and vowel order.

    Args:
        grapheme: One grapheme cluster

    Returns:
        (base character or the grapheme itself, vowel order or VOWEL_NONE)
    """
    if not grapheme:
        return UNK_TOKEN, VOWEL_NONE
    code = ord(grapheme[0])
    if SYLLABARY_START <= code <= SYLLABARY_END and len(grapheme) == 1:
        offset = code - SYLLABARY_START
        return chr(SYLLABARY_START + (offset // 8) * 8), offset % 8
    return grapheme, VOWEL_NONE


def compose_fidel(base: str, vowel: int) -> str:
    """Inverse of decompose_fidel."""
    if vowel == VOWEL_NONE or not base:
        return base
    code = ord(base[0])
    if SYLLABARY_START <= code <= SYLLABARY_END:
        return chr(code + vowel)
    return base


class FidelVocab:
    """Maps graphemes to (base, vowel) index pairs and back."""

    def __init__(self, bases: list[str]):
        self.bases = list(bases)
        self.base_to_idx = {b: i for i, b in enumerate(self.bases)}
        self.pad_idx = self.base_to_idx[PAD_TOKEN]
        self.missing_idx = self.base_to_idx[MISSING_MARKER]
        self.gap_idx = self.base_to_idx[UNKNOWN_GAP_MARKER]
        self.unk_idx = self.base_to_idx[UNK_TOKEN]
        self.nofill_idx = self.base_to_idx[NOFILL_MARKER]

    def encode_tokens(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        """
        Encode pre-split tokens, which may include markers.

        Args:
            tokens: Grapheme clusters and/or damage markers

        Returns:
            (base indices, vowel indices)
        """
        base_ids: list[int] = []
        vowel_ids: list[int] = []
        for token in tokens:
            if token in SPECIAL_BASES:
                base_ids.append(self.base_to_idx[token])
                vowel_ids.append(VOWEL_NONE)
                continue
            base, vowel = decompose_fidel(token)
            base_ids.append(self.base_to_idx.get(base, self.unk_idx))
            vowel_ids.append(vowel)
        return base_ids, vowel_ids

    @classmethod
    def from_corpus(cls, texts: list[str], min_count: int = 1) -> "FidelVocab":
        """
        Build a vocabulary from raw text.

        Args:
            texts: Corpus strings
            min_count: Drop bases rarer than this

        Returns:
            FidelVocab
        """
        counts: dict[str, int] = {}
        for text in texts:
            for grapheme in split_graphemes(text):
                base, _ = decompose_fidel(grapheme)
                counts[base] = counts.get(base, 0) + 1

        kept = sorted(b for b, n in counts.items()
                      if n >= min_count and b not in SPECIAL_BASES)
        return cls(list(SPECIAL_BASES) + kept)

    def __len__(self) -> int:
        return len(self.bases)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """
        Encode text to parallel base and vowel index lists.

        Args:
            text: Possibly damaged text containing gap markers

        Returns:
            (base indices, vowel indices)
        """
        base_ids: list[int] = []
        vowel_ids: list[int] = []
        for token in self._tokenise(text):
            if token in (MISSING_MARKER, UNKNOWN_GAP_MARKER):
                base_ids.append(self.base_to_idx[token])
                vowel_ids.append(VOWEL_NONE)
                continue
            base, vowel = decompose_fidel(token)
            base_ids.append(self.base_to_idx.get(base, self.unk_idx))
            vowel_ids.append(vowel)
        return base_ids, vowel_ids

    def decode(self, base_ids: list[int], vowel_ids: list[int]) -> str:
        """Rebuild a string from index pairs."""
        out: list[str] = []
        for b, v in zip(base_ids, vowel_ids):
            if b == self.pad_idx:
                continue
            base = self.bases[b] if 0 <= b < len(self.bases) else UNK_TOKEN
            out.append(base if base in SPECIAL_BASES else compose_fidel(base, v))
        return "".join(out)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Split text into graphemes, keeping gap markers intact."""
        tokens: list[str] = []
        remainder = text
        while remainder:
            hit = None
            for marker in (MISSING_MARKER, UNKNOWN_GAP_MARKER):
                index = remainder.find(marker)
                if index != -1 and (hit is None or index < hit[0]):
                    hit = (index, marker)
            if hit is None:
                tokens.extend(split_graphemes(remainder))
                break
            index, marker = hit
            if index:
                tokens.extend(split_graphemes(remainder[:index]))
            tokens.append(marker)
            remainder = remainder[index + len(marker):]
        return tokens

    def fingerprint(self) -> str:
        """Stable hash so checkpoints cannot load against a changed vocabulary."""
        return hashlib.sha256("".join(self.bases).encode("utf-8")).hexdigest()[:16]


@dataclass
class RestorationConfig:
    """Model hyperparameters."""

    vocab_size: int
    emb_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 1024
    max_len: int = 256
    dropout: float = 0.1


class GeezRestorationModel(nn.Module):
    """Transformer encoder predicting base and vowel at every position."""

    def __init__(self, config: RestorationConfig):
        super().__init__()
        self.config = config

        self.base_emb = nn.Embedding(config.vocab_size, config.emb_dim,
                                     padding_idx=0)
        self.vowel_emb = nn.Embedding(NUM_VOWEL_CLASSES, config.emb_dim)
        self.pos_emb = nn.Embedding(config.max_len, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.emb_dim)

        self.base_head = nn.Linear(config.emb_dim, config.vocab_size)
        self.vowel_head = nn.Linear(config.emb_dim, NUM_VOWEL_CLASSES)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(
        self,
        base_ids: torch.Tensor,
        vowel_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_ids: (batch, seq) base indices
            vowel_ids: (batch, seq) vowel indices
            pad_mask: (batch, seq) True where padded

        Returns:
            (base logits, vowel logits)
        """
        seq_len = base_ids.size(1)
        positions = torch.arange(seq_len, device=base_ids.device)

        x = (self.base_emb(base_ids)
             + self.vowel_emb(vowel_ids)
             + self.pos_emb(positions).unsqueeze(0))
        x = self.dropout(x * math.sqrt(self.config.emb_dim))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return self.base_head(x), self.vowel_head(x)

    def parameter_count(self) -> int:
        """Trainable parameter total."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_restoration_model(model: GeezRestorationModel, vocab: FidelVocab,
                           path: Path) -> None:
    """Persist weights alongside the vocabulary that produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
        "bases": vocab.bases,
        "vocab_fingerprint": vocab.fingerprint(),
    }, path)


def load_restoration_model(path: Path) -> tuple[GeezRestorationModel, FidelVocab]:
    """Load a checkpoint, refusing to run against a mismatched vocabulary."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocab = FidelVocab(checkpoint["bases"])
    if vocab.fingerprint() != checkpoint.get("vocab_fingerprint"):
        raise ValueError("Vocabulary fingerprint mismatch in checkpoint")
    model = GeezRestorationModel(RestorationConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab
