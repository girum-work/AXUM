from __future__ import annotations

import json
import csv

import pytest
from PIL import Image

from src.ocr.training_contract import (
    OCR_WORD_SEPARATOR,
    audit_labels,
    build_assigned_ethiopic_charset,
    compute_sequence_metrics,
    label_fits_ctc,
    levenshtein_distance,
    load_class_map,
    min_ctc_timesteps,
    normalize_ocr_label,
)
from src.ocr.pipeline import HHDEthiopicDataset


def test_normalize_label_uses_nfc_and_ethiopic_word_separator() -> None:
    assert normalize_ocr_label("  ሰላም   ዓለም  ") == f"ሰላም{OCR_WORD_SEPARATOR}ዓለም"


def test_ctc_budget_accounts_for_adjacent_repeats() -> None:
    assert min_ctc_timesteps("ሀሀሁ") == 4
    assert label_fits_ctc("ሀ" * 14, timesteps=26) is False
    assert label_fits_ctc("ሀሁ" * 13, timesteps=26) is True


def test_edit_distance_metrics_count_insertions_and_deletions() -> None:
    assert levenshtein_distance("ሰላም", "ሰም") == 1
    metrics = compute_sequence_metrics(["ሰላም", "ሀ"], ["ሰም", "ሀሁ"])
    assert metrics.samples == 2
    assert metrics.reference_characters == 4
    assert metrics.edit_distance == 2
    assert metrics.cer == pytest.approx(0.5)
    assert metrics.character_accuracy == pytest.approx(0.5)
    assert metrics.sequence_accuracy == 0.0


def test_charset_excludes_unassigned_codepoints() -> None:
    charset = build_assigned_ethiopic_charset()
    assert "ሀ" in charset
    assert "\u1249" not in charset


def test_audit_rejects_unassigned_and_unknown_labels() -> None:
    charset = build_assigned_ethiopic_charset()
    audit = audit_labels(["ሀ", "\u1249", "A"], charset)
    assert audit.valid is False
    assert audit.unassigned_characters == {"\u1249": 1}
    assert audit.unknown_characters == {"A": 1}


def test_class_map_must_be_explicit_assigned_and_unique(tmp_path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"0": "ሀ", "1": "ሁ"}, ensure_ascii=False), encoding="utf-8")
    assert load_class_map(valid) == {0: "ሀ", 1: "ሁ"}

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"0": "\u1249"}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid class mapping"):
        load_class_map(invalid)


def test_explicit_manifest_normalizes_and_rejects_invalid_labels(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("valid.png", "unassigned.png", "too_long.png"):
        Image.new("L", (64, 32), 255).save(image_dir / name)
    manifest = tmp_path / "labels.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([
            ("valid.png", "ሰላም ዓለም"),
            ("unassigned.png", "\u1249"),
            ("too_long.png", "ሀ" * 30),  # exceeds OCR_CTC_SEQ_LEN (58) with repeat blanks
        ])

    dataset = HHDEthiopicDataset(
        str(tmp_path),
        split="test",
        augment=False,
        manifest_path=manifest,
        image_dir=image_dir,
    )
    assert len(dataset) == 1
    assert dataset.samples[0][1] == f"ሰላም{OCR_WORD_SEPARATOR}ዓለም"
