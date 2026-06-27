#!/usr/bin/env python3
"""
AXUM Rover — Conservation Knowledge Base Builder
=================================================
Merges fragment JSON files from data/databases/kb/ into a single
conservation_kb.json consumed by src/analysis/treatment_advisor.py.

The compatibility matrix combines:
  1. Hand-authored explicit entries (compatibility_matrix_explicit.json)
  2. Programmatic entries derived from treatment metadata for every
     substrate+decay pair where decay.affected_substrates includes the substrate.

Run from project root:
    python scripts/build_conservation_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_FRAG_DIR = PROJECT_ROOT / "data" / "databases" / "kb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "databases" / "conservation_kb.json"

# Substrates that require hand-authored matrix entries (minimum coverage)
EXPLICIT_SUBSTRATE_IDS = [
    "limestone_porous",
    "marble",
    "sandstone",
    "basalt",
    "tuff_ignimbrite",
    "terracotta_ceramic",
    "parchment_vellum",
    "painted_surface_fresco",
    "metal_bronze",
]


def _load_fragment(name: str) -> dict:
    """
    Load a single KB fragment JSON file.

    Args:
        name: Filename without path (e.g. 'substrates.json')

    Returns:
        Parsed JSON dict; empty dict if file missing or empty
    """
    path = KB_FRAG_DIR / name
    if not path.exists():
        logger.warning(f"Fragment missing: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _treatment_mechanism(treatment: dict, substrate_id: str, decay_id: str) -> str:
    """
    Build a human-readable incompatibility mechanism string.

    Args:
        treatment: Treatment definition from KB
        substrate_id: Substrate being evaluated
        decay_id: Decay pattern being evaluated

    Returns:
        Explanation string for unsafe list entries
    """
    if treatment.get("danger_reason"):
        return treatment["danger_reason"]
    if substrate_id in treatment.get("never_use_on", []):
        return treatment.get(
            "incompatibility_reason",
            f"Listed in never_use_on for {substrate_id}",
        )
    if substrate_id in treatment.get("incompatible_substrates", []):
        return treatment.get(
            "incompatibility_reason",
            f"Incompatible with substrate {substrate_id}",
        )
    if decay_id in treatment.get("incompatible_with_decay", []):
        return treatment.get(
            "incompatibility_reason_decay",
            f"Incompatible with decay process {decay_id}",
        )
    return treatment.get("incompatibility_reason", "Compatibility rule exclusion")


def _infer_matrix_entry(
    substrate_id: str,
    decay_id: str,
    treatments: dict,
) -> dict:
    """
    Derive safe/unsafe treatment lists from treatment-level metadata.

    Args:
        substrate_id: KB substrate ID
        decay_id: KB decay pattern ID
        treatments: Full treatments dict from KB

    Returns:
        Matrix entry with safe, unsafe, sequence, safe_after_desalination keys
    """
    safe_ids: list[str] = []
    unsafe: list[dict] = []

    for tid, t in treatments.items():
        ttype = t.get("type", "")
        compat_sub = t.get("compatible_substrates", [])
        compat_decay = t.get("compatible_decay", [])
        never_on = t.get("never_use_on", [])
        incompat_sub = t.get("incompatible_substrates", [])
        incompat_decay = t.get("incompatible_with_decay", [])

        # Hard exclusions
        if substrate_id in never_on or substrate_id in incompat_sub:
            unsafe.append({
                "treatment": tid,
                "risk": t.get("danger_level", "high"),
                "mechanism": _treatment_mechanism(t, substrate_id, decay_id),
                "source": t.get("source", ""),
            })
            continue

        if decay_id in incompat_decay:
            unsafe.append({
                "treatment": tid,
                "risk": "high",
                "mechanism": _treatment_mechanism(t, substrate_id, decay_id),
                "source": t.get("source", ""),
            })
            continue

        # Dangerous treatments — flag unless explicitly excluded from never_use_on
        if ttype == "DANGEROUS_TREATMENT":
            unsafe.append({
                "treatment": tid,
                "risk": t.get("danger_level", "critical"),
                "mechanism": t.get("danger_reason", ""),
                "source": t.get("source", ""),
            })
            continue

        sub_ok = not compat_sub or substrate_id in compat_sub
        decay_ok = not compat_decay or decay_id in compat_decay
        if sub_ok and decay_ok:
            safe_ids.append(tid)

    sequence = ""
    if decay_id in ("salt_crystallization", "subflorescence", "efflorescence"):
        sequence = (
            "Document → isolate moisture source → desalination poultice cycles "
            "→ monitor conductivity → consolidant only after salt equilibrium"
        )
    elif decay_id == "biological_colonization":
        sequence = "Dry-brush/soil removal → biocide (if warranted) → rinse → monitor"
    elif decay_id == "bronze_disease":
        sequence = "Document → mechanical cleaning → BTA inhibitor → RH control → wax barrier"

    safe_after = []
    if decay_id in ("salt_crystallization", "subflorescence", "efflorescence"):
        for tid in safe_ids:
            t = treatments.get(tid, {})
            if t.get("type") in ("CONSOLIDANT", "COATING"):
                safe_after.append(tid)

    return {
        "safe": safe_ids,
        "unsafe": unsafe,
        "sequence": sequence,
        "safe_after_desalination": safe_after,
    }


def _merge_matrix_entries(base: dict, override: dict) -> dict:
    """
    Merge programmatic matrix entry with explicit hand-authored entry.

    Explicit entries take precedence for sequence and safe_after_desalination.
    Safe/unsafe lists are unioned with explicit unsafe taking priority.

    Args:
        base: Programmatically generated entry
        override: Hand-authored explicit entry

    Returns:
        Merged matrix entry
    """
    merged = dict(base)
    if override.get("sequence"):
        merged["sequence"] = override["sequence"]
    if override.get("safe_after_desalination"):
        merged["safe_after_desalination"] = override["safe_after_desalination"]

    explicit_safe = set(override.get("safe", []))
    explicit_unsafe_ids = {u["treatment"] for u in override.get("unsafe", [])}

    safe = list(dict.fromkeys(override.get("safe", []) + [
        s for s in base.get("safe", []) if s not in explicit_unsafe_ids
    ]))
    unsafe_by_id = {u["treatment"]: u for u in base.get("unsafe", [])}
    for u in override.get("unsafe", []):
        unsafe_by_id[u["treatment"]] = u

    merged["safe"] = [s for s in safe if s not in explicit_unsafe_ids]
    merged["unsafe"] = list(unsafe_by_id.values())
    return merged


def build_compatibility_matrix(
    substrates: dict,
    decay_patterns: dict,
    treatments: dict,
    explicit: dict,
) -> dict:
    """
    Build full compatibility matrix from explicit + programmatic rules.

    Args:
        substrates: Substrate definitions keyed by ID
        decay_patterns: Decay pattern definitions keyed by ID
        treatments: Treatment definitions keyed by ID
        explicit: Hand-authored matrix fragment (substrate → decay → entry)

    Returns:
        Nested dict: substrate_id → decay_id → compatibility entry
    """
    matrix: dict = {}

    for substrate_id in substrates:
        matrix[substrate_id] = {}
        for decay_id, decay in decay_patterns.items():
            affected = decay.get("affected_substrates", [])
            if affected and substrate_id not in affected:
                continue

            inferred = _infer_matrix_entry(substrate_id, decay_id, treatments)
            override = explicit.get(substrate_id, {}).get(decay_id, {})
            if override:
                matrix[substrate_id][decay_id] = _merge_matrix_entries(inferred, override)
            else:
                matrix[substrate_id][decay_id] = inferred

    # Ensure all explicit combos exist even if affected_substrates omitted
    for substrate_id, decays in explicit.items():
        if substrate_id not in matrix:
            matrix[substrate_id] = {}
        for decay_id, entry in decays.items():
            if decay_id not in matrix[substrate_id]:
                inferred = _infer_matrix_entry(substrate_id, decay_id, treatments)
                matrix[substrate_id][decay_id] = _merge_matrix_entries(inferred, entry)
            elif not matrix[substrate_id][decay_id].get("sequence") and entry.get("sequence"):
                matrix[substrate_id][decay_id] = _merge_matrix_entries(
                    matrix[substrate_id][decay_id], entry
                )

    return matrix


def count_matrix_entries(matrix: dict) -> int:
    """Count total substrate+decay compatibility entries in matrix."""
    return sum(len(decays) for decays in matrix.values())


def build_kb() -> dict:
    """
    Assemble complete conservation knowledge base from fragments.

    Returns:
        Merged KB dict ready for JSON serialisation
    """
    metadata = _load_fragment("metadata.json")
    substrates = _load_fragment("substrates.json")
    decay_patterns = _load_fragment("decay_patterns.json")
    treatments = _load_fragment("treatments.json")
    urgency_levels = _load_fragment("urgency_levels.json")
    artefact_substrate_map = _load_fragment("artefact_substrate_map.json")
    explicit_matrix = _load_fragment("compatibility_matrix_explicit.json")

    compatibility_matrix = build_compatibility_matrix(
        substrates, decay_patterns, treatments, explicit_matrix
    )

    return {
        "metadata": metadata,
        "substrates": substrates,
        "decay_patterns": decay_patterns,
        "treatments": treatments,
        "compatibility_matrix": compatibility_matrix,
        "urgency_levels": urgency_levels,
        "artefact_substrate_map": artefact_substrate_map,
    }


def main() -> int:
    """Build and write conservation_kb.json; print summary statistics."""
    logger.info(f"Building conservation KB from {KB_FRAG_DIR}")

    kb = build_kb()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    matrix_count = count_matrix_entries(kb["compatibility_matrix"])
    output_size = OUTPUT_PATH.stat().st_size

    # Fragment sizes
    fragment_sizes = {}
    for frag in KB_FRAG_DIR.glob("*.json"):
        fragment_sizes[frag.name] = frag.stat().st_size

    summary = {
        "output_path": str(OUTPUT_PATH),
        "output_size_bytes": output_size,
        "fragment_sizes_bytes": fragment_sizes,
        "substrate_count": len(kb["substrates"]),
        "decay_count": len(kb["decay_patterns"]),
        "treatment_count": len(kb["treatments"]),
        "matrix_entry_count": matrix_count,
        "urgency_levels": len(kb["urgency_levels"]),
    }

    print(json.dumps(summary, indent=2))
    logger.success(
        f"Wrote {OUTPUT_PATH} — "
        f"{summary['substrate_count']} substrates, "
        f"{summary['decay_count']} decay patterns, "
        f"{summary['treatment_count']} treatments, "
        f"{matrix_count} matrix entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
