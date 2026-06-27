"""
AXUM ROVER — Ge'ez inscription text corpus
=============================================
Unified clean-text sources for restoration fine-tuning and few-shot
pattern matching. Curated epigraphic lines + optional HHD-Ethiopic labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from config import (
    GEEZ_RESTORATION_CORPUS,
    HHD_MERGED_CSV,
    HHD_FALLBACK_CSV,
    RESTORATION_HHD_MAX_PHRASES,
    RESTORATION_HHD_MIN_GRAPHEMES,
    RESTORATION_HHD_MAX_GRAPHEMES,
)
from src.ocr.damage import is_valid_geez_text


@dataclass
class InscriptionEntry:
    """
    One clean inscription line with metadata.

    Attributes:
        id: Stable phrase ID for train/eval splits
        text: Clean Ge'ez string
        translation: English gloss (may be placeholder for HHD-sourced lines)
        period: Historical period string
        location: Site or corpus provenance
        context: Epigraphic notes for reasoning field
        source: 'curated' | 'hhd' | 'user'
    """

    id: str
    text: str
    translation: str
    period: str
    location: str
    context: str
    source: str = "curated"


# Curated epigraphic corpus (Aksumite, Zagwe, liturgical)
CURATED_ENTRIES: list[InscriptionEntry] = [
    InscriptionEntry(
        "AKS_001", "ሰላም", "Peace / Greeting",
        "All periods", "Throughout Ethiopia",
        "Most frequent single-word inscription in Ethiopian heritage sites",
    ),
    InscriptionEntry(
        "AKS_002", "ሰላም ለኢትዮጵያ", "Peace to Ethiopia",
        "Aksumite, 4th century CE", "Aksum",
        "National dedicatory formula",
    ),
    InscriptionEntry(
        "AKS_003", "ዓጼ ዓዛና ነጉሠ ነገሥት", "Emperor Ezana, King of Kings",
        "Aksumite, 4th century CE", "Aksum stelae field",
        "Royal title of King Ezana",
    ),
    InscriptionEntry(
        "AKS_004", "ዓጼ ካሌብ ነጉሠ ነገሥት ዘኢትዮጵያ",
        "Emperor Kaleb, King of Kings of Ethiopia",
        "Aksumite, 6th century CE", "Aksum",
        "King Kaleb inscription",
    ),
    InscriptionEntry(
        "AKS_005", "ነጉሥ ዘኢትዮጵያ ወ ዘሕዝብ", "King of Ethiopia and the People",
        "Aksumite", "Royal inscription sites", "Standard royal formula",
    ),
    InscriptionEntry(
        "AKS_006", "ዘወርቅ ወዘብሩር", "Of gold and of silver",
        "Aksumite, 3rd-6th CE", "Aksum coin inscriptions",
        "Aksumite coin dedication formula",
    ),
    InscriptionEntry(
        "AKS_007", "ጽዮን ታቦተ ሕጉ", "Zion, the Ark of the Law",
        "Aksumite+", "Aksum, Maryam Tsion church",
        "Reference to the Ark of the Covenant",
    ),
    InscriptionEntry(
        "ZAG_001", "ቅዱስ ጊዮርጊስ", "Saint George",
        "Post-5th century CE", "Ethiopian churches",
        "Most common named saint in Ethiopian inscriptions",
    ),
    InscriptionEntry(
        "ZAG_002", "ማርያም ወወልድ", "Mary and the Son (of God)",
        "Zagwe, 12th century CE", "Lalibela, Beta Maryam",
        "Beta Maryam church dedication",
    ),
    InscriptionEntry(
        "ZAG_003", "ቤተ ማርያም ዘተሐነፀ", "The House of Mary that was built",
        "Zagwe, 12th century CE", "Lalibela", "Church foundation inscription",
    ),
    InscriptionEntry(
        "ZAG_004", "ክርስቶስ አምላክ ወወልድ", "Christ, God and Son",
        "Zagwe, 12th century CE", "Lalibela churches",
        "Christological formula common in Zagwe period",
    ),
    InscriptionEntry(
        "ZAG_005", "ቅዱስ ሚካኤል ሊቀ መልአክ", "Saint Michael, Archangel",
        "Post-5th century CE", "Beta Mika'el church, Lalibela",
        "Archangel Michael dedication",
    ),
    InscriptionEntry(
        "LIT_001", "ሐሌሉያ ለእግዚአብሔር", "Hallelujah to God",
        "All Christian periods", "Ethiopian churches", "Liturgical praise",
    ),
    InscriptionEntry(
        "LIT_002", "አምላከ ሰማይ ወምድር", "God of Heaven and Earth",
        "Post-4th century CE", "Church inscriptions", "Theological formula",
    ),
    InscriptionEntry(
        "LIT_003", "አብ ወወልድ ወመንፈስ ቅዱስ", "Father and Son and Holy Spirit",
        "Post-4th century CE", "Trinitarian inscriptions",
        "Full Trinitarian formula",
    ),
    InscriptionEntry(
        "LIT_004", "ወልደ እግዚአብሔር", "Son of God",
        "Zagwe, 12th century CE", "Lalibela", "Christological title",
    ),
    InscriptionEntry(
        "LIT_005", "ሃይማኖት ወሰላም", "Faith and Peace",
        "Post-4th century CE", "Church inscriptions",
        "Combined theological dedication",
    ),
    InscriptionEntry(
        "LIT_006", "ቅዱስ ያሬድ ዘዜማ", "Saint Yared of Music",
        "Aksumite, 6th century CE", "Aksum area",
        "Yared — Ethiopian Orthodox hymnography",
    ),
    InscriptionEntry(
        "AKS_008", "ኢትዮጵያ ትቀድም ኢዮሐ",
        "Ethiopia, precede in faith",
        "Aksumite+", "Royal inscriptions",
        "Derived from Psalms 68:31",
    ),
    InscriptionEntry(
        "LIT_007", "ለእግዚአብሔር ክብር", "Glory to God",
        "All Christian periods", "Church inscriptions",
        "Universal Christian dedication",
    ),
    InscriptionEntry(
        "ZAG_006", "ዮሐንስ ሐዋርያ", "John the Apostle",
        "Post-4th century CE", "Beta Iyesus, Lalibela", "Apostle John dedication",
    ),
    InscriptionEntry(
        "ZAG_007", "ጊዮርጊስ ሰማዕት", "George the Martyr",
        "Post-5th century CE", "Beta Giyorgis, Lalibela",
        "Saint George as martyr",
    ),
    InscriptionEntry(
        "AKS_009", "አክሱም ዘሀገረ ጽዮን", "Aksum, City of Zion",
        "Aksumite", "Aksum", "Aksum sacred identity",
    ),
    InscriptionEntry(
        "LIT_008", "ፍቅር ወሰላም ወጽድቅ", "Love and Peace and Righteousness",
        "All periods", "Church inscriptions", "Three-virtue dedication",
    ),
    InscriptionEntry(
        "LIT_009", "ነቢይ ኤልያስ", "Prophet Elijah",
        "Post-4th century CE", "Ethiopian church inscriptions",
        "Old Testament prophet venerated in Ethiopian tradition",
    ),
]


def _phrase_id_for_text(text: str, prefix: str = "HHD") -> str:
    """Stable short ID from text content."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def normalize_hhd_label(label: str) -> str:
    """
    Normalize HHD CSV labels for restoration corpus.

    Replaces Ethiopic word space (U+1361) with ASCII space and strips
    trailing word-space punctuation.

    Args:
        label: Raw CSV text field

    Returns:
        Normalized Ge'ez string
    """
    text = label.strip()
    text = text.replace("\u1361", " ")
    text = text.replace("፡", " ")
    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def load_hhd_phrases(
    csv_path: Path | None = None,
    max_phrases: int | None = None,
    min_graphemes: int | None = None,
    max_graphemes: int | None = None,
) -> list[InscriptionEntry]:
    """
    Load unique clean text lines from HHD-Ethiopic image_text_pairs CSV.

    Args:
        csv_path: Path to image_text_pairs_train.csv
        max_phrases: Cap on unique phrases imported
        min_graphemes: Minimum content length filter
        max_graphemes: Maximum content length filter

    Returns:
        List of InscriptionEntry with source='hhd'
    """
    if csv_path is not None:
        path = Path(csv_path)
    else:
        path = None
        for candidate in (HHD_MERGED_CSV, HHD_FALLBACK_CSV):
            if candidate.exists():
                path = candidate
                break
        if path is None:
            path = HHD_MERGED_CSV

    max_phrases = max_phrases if max_phrases is not None else RESTORATION_HHD_MAX_PHRASES
    min_g = min_graphemes if min_graphemes is not None else RESTORATION_HHD_MIN_GRAPHEMES
    max_g = max_graphemes if max_graphemes is not None else RESTORATION_HHD_MAX_GRAPHEMES

    if not path.exists():
        logger.warning(f"HHD CSV not found: {path}")
        return []

    seen: set[str] = set()
    entries: list[InscriptionEntry] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            raw = row[1].strip()
            text = normalize_hhd_label(raw)
            if text in seen:
                continue
            if not is_valid_geez_text(text, min_len=min_g, max_len=max_g):
                continue
            seen.add(text)
            entries.append(
                InscriptionEntry(
                    id=_phrase_id_for_text(text),
                    text=text,
                    translation="[Manuscript line — verify against source edition]",
                    period="Manuscript / printed Ethiopic, 18th–20th c.",
                    location="HHD-Ethiopic corpus",
                    context="Word-level label from HHD-Ethiopic OCR training set",
                    source="hhd",
                )
            )
            if len(entries) >= max_phrases:
                break

    logger.info(f"Loaded {len(entries)} unique HHD phrases from {path.name}")
    return entries


def load_corpus_json(path: Path | None = None) -> list[InscriptionEntry]:
    """
    Load extra entries from JSONL corpus file (one JSON object per line).

    Args:
        path: Defaults to GEEZ_RESTORATION_CORPUS

    Returns:
        Parsed InscriptionEntry list
    """
    path = Path(path or GEEZ_RESTORATION_CORPUS)
    if not path.exists():
        return []

    entries: list[InscriptionEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                InscriptionEntry(
                    id=data.get("id", _phrase_id_for_text(data["text"], "USR")),
                    text=data["text"],
                    translation=data.get("translation", ""),
                    period=data.get("period", "Unknown"),
                    location=data.get("location", "Unknown"),
                    context=data.get("context", ""),
                    source=data.get("source", "user"),
                )
            )
    return entries


def build_full_corpus(
    include_hhd: bool = True,
    include_json: bool = True,
) -> list[InscriptionEntry]:
    """
    Merge curated, JSONL, and optional HHD sources (dedupe by text).

    Args:
        include_hhd: Import from HHD CSV when available
        include_json: Load GEEZ_RESTORATION_CORPUS JSONL

    Returns:
        Deduplicated list of InscriptionEntry
    """
    by_text: dict[str, InscriptionEntry] = {}

    for entry in CURATED_ENTRIES:
        by_text[entry.text] = entry

    if include_json:
        for entry in load_corpus_json():
            if entry.text not in by_text:
                by_text[entry.text] = entry

    if include_hhd:
        for entry in load_hhd_phrases():
            if entry.text not in by_text:
                by_text[entry.text] = entry

    return list(by_text.values())


def save_corpus_json(entries: list[InscriptionEntry], path: Path | None = None) -> Path:
    """
    Export corpus to JSONL for manual editing.

    Args:
        entries: Full corpus to write
        path: Output path (default GEEZ_RESTORATION_CORPUS)

    Returns:
        Path written
    """
    path = Path(path or GEEZ_RESTORATION_CORPUS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(entries)} corpus lines to {path}")
    return path


def split_phrase_ids(
    entries: list[InscriptionEntry],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """
    Split phrase IDs into train/val/test (no phrase leakage across splits).

    Args:
        entries: Full corpus
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = remainder)
        seed: RNG seed for shuffle

    Returns:
        (train_ids, val_ids, test_ids)
    """
    import random

    ids = sorted({e.id for e in entries})
    random.seed(seed)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    return train_ids, val_ids, test_ids
