"""Shared data and evaluation contracts for AXUM OCR experiments."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


OCR_CTC_TIMESTEPS = 58
OCR_BLANK_TOKEN = "<BLANK>"
OCR_UNKNOWN_TOKEN = "<UNK>"
OCR_WORD_SEPARATOR = "፡"


def normalize_ocr_label(text: str) -> str:
    """Normalize OCR labels to NFC and one canonical Ethiopic word separator."""
    normalized = unicodedata.normalize("NFC", text.strip())
    normalized = " ".join(normalized.split())
    return normalized.replace(" ", OCR_WORD_SEPARATOR)


def min_ctc_timesteps(label: str) -> int:
    """Return exact CTC steps required, including blanks between repeated symbols."""
    repeats = sum(left == right for left, right in zip(label, label[1:]))
    return len(label) + repeats


def label_fits_ctc(label: str, timesteps: int = OCR_CTC_TIMESTEPS) -> bool:
    return bool(label) and min_ctc_timesteps(label) <= timesteps


def is_assigned_character(character: str) -> bool:
    return len(character) == 1 and unicodedata.category(character) != "Cn"


def build_assigned_ethiopic_charset(extra_characters: Iterable[str] = ()) -> tuple[str, ...]:
    """Build a stable charset from assigned Ethiopic code points only."""
    characters = [
        chr(codepoint)
        for codepoint in range(0x1200, 0x1380)
        if is_assigned_character(chr(codepoint))
    ]
    for character in extra_characters:
        if is_assigned_character(character) and character not in characters:
            characters.append(character)
    return tuple([OCR_BLANK_TOKEN, *characters, OCR_UNKNOWN_TOKEN])


def charset_fingerprint(charset: Sequence[str]) -> str:
    payload = json.dumps(list(charset), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Compute Unicode code-point edit distance using linear memory."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, pred_char in enumerate(prediction, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_char != pred_char),
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class SequenceMetrics:
    samples: int
    reference_characters: int
    edit_distance: int
    exact_matches: int

    @property
    def cer(self) -> float:
        return self.edit_distance / max(1, self.reference_characters)

    @property
    def character_accuracy(self) -> float:
        return max(0.0, 1.0 - self.cer)

    @property
    def sequence_accuracy(self) -> float:
        return self.exact_matches / max(1, self.samples)

    def to_dict(self) -> dict[str, float | int]:
        payload = asdict(self)
        payload.update({
            "cer": self.cer,
            "character_accuracy": self.character_accuracy,
            "sequence_accuracy": self.sequence_accuracy,
        })
        return payload


def compute_sequence_metrics(references: Sequence[str], predictions: Sequence[str]) -> SequenceMetrics:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal lengths")
    normalized_references = [normalize_ocr_label(text) for text in references]
    normalized_predictions = [normalize_ocr_label(text) for text in predictions]
    return SequenceMetrics(
        samples=len(normalized_references),
        reference_characters=sum(len(text) for text in normalized_references),
        edit_distance=sum(
            levenshtein_distance(reference, prediction)
            for reference, prediction in zip(normalized_references, normalized_predictions)
        ),
        exact_matches=sum(
            reference == prediction
            for reference, prediction in zip(normalized_references, normalized_predictions)
        ),
    )


@dataclass(frozen=True)
class LabelAudit:
    total: int
    empty: int
    ctc_infeasible: int
    unassigned_characters: dict[str, int]
    unknown_characters: dict[str, int]

    @property
    def valid(self) -> bool:
        return self.empty == 0 and self.ctc_infeasible == 0 and not self.unassigned_characters and not self.unknown_characters


def audit_labels(labels: Iterable[str], charset: Sequence[str]) -> LabelAudit:
    allowed = set(charset) - {OCR_BLANK_TOKEN, OCR_UNKNOWN_TOKEN}
    total = empty = infeasible = 0
    unassigned: dict[str, int] = {}
    unknown: dict[str, int] = {}
    for raw_label in labels:
        total += 1
        label = normalize_ocr_label(raw_label)
        if not label:
            empty += 1
            continue
        if not label_fits_ctc(label):
            infeasible += 1
        for character in label:
            if not is_assigned_character(character):
                unassigned[character] = unassigned.get(character, 0) + 1
            elif character not in allowed:
                unknown[character] = unknown.get(character, 0) + 1
    return LabelAudit(total, empty, infeasible, unassigned, unknown)


def load_class_map(path: str | Path) -> dict[int, str]:
    """Load an explicit JSON class-index map and reject ambiguous labels."""
    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        mapping = {index: value for index, value in enumerate(payload)}
    elif isinstance(payload, dict):
        mapping = {int(index): value for index, value in payload.items()}
    else:
        raise ValueError("class map must be a JSON list or object")

    result: dict[int, str] = {}
    for index, raw_character in mapping.items():
        character = normalize_ocr_label(str(raw_character))
        if len(character) != 1 or not is_assigned_character(character):
            raise ValueError(f"invalid class mapping {index!r} -> {raw_character!r}")
        result[index] = character
    if len(set(result.values())) != len(result):
        raise ValueError("class map contains duplicate characters")
    return result
