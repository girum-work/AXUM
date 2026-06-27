# src/analysis/treatment_advisor.py
"""
AXUM ROVER — Treatment Compatibility Advisor
=============================================
Decision guardrail that converts sensor pipeline outputs into field-ready
treatment protocols: what is SAFE to apply, what is DANGEROUS, and why.

Addresses the Fidler (2005) gap: ~60% of conservation work is done by
untrained practitioners who can increase damage up to 50%. This module
automates the "thorough study" phase so inexpert field workers follow
validated instructions instead of guessing.

INPUTS (from robot sensing pipeline):
    - Artefact class (MobileNetV2 / YOLO classifier)
    - Crack severity (crack detector)
    - Salt risk (salt_mapper UV fluorescence + conductivity)
    - Surface stress (multispectral NDCI)
    - OCR confidence (inscription legibility / biological crust proxy)
    - Material hardness (photometric stereo curvature, optional)

OUTPUTS:
    - Ranked safe treatments with application instructions
    - Explicit dangerous treatment warnings with harm mechanisms
    - Treatment sequence (order of operations)
    - Urgency level (monitor → emergency)
    - Overall intervention risk score

Author: Axum Rover Team
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CONSERVATION_KB_PATH,
    ARTEFACT_SUBSTRATE_MAP,
    TREATMENT_URGENCY_CRITICAL_YEARS,
    TREATMENT_URGENCY_PRIORITY_YEARS,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DiagnosticInputs:
    """
    Sensor outputs fed into the treatment advisor.

    All fields have safe defaults so the module can run with partial
    pipeline data (graceful degradation per project rules).

    Attributes:
        artefact_id:         Catalogue identifier e.g. AX002
        artefact_class:      Classifier label (pottery, stone_carving, etc.)
        substrate_override:  Force a specific substrate ID from KB
        crack_severity:      0–1 from crack detector
        salt_risk:           0–1 from salt mapper
        salt_critical:       True if salt risk level ≥ 3
        stress_score:        0–1 from multispectral NDCI
        ocr_confidence:      0–1 from Ge'ez OCR pipeline
        has_inscription:     True if OCR found any text
        hardness_score:      0–1 normalised surface hardness (optional)
        biological_detected: True if multispectral bio classifier fired
        years_remaining:     Fragility clock estimate (optional)
        active_moisture:     True if moisture source still active on site
    """
    artefact_id:         str   = "UNKNOWN"
    artefact_class:      str   = "other"
    substrate_override:  Optional[str] = None
    crack_severity:      float = 0.0
    salt_risk:           float = 0.0
    salt_critical:       bool  = False
    stress_score:        float = 0.0
    ocr_confidence:      float = 1.0
    has_inscription:     bool  = False
    hardness_score:      Optional[float] = None
    biological_detected: bool  = False
    years_remaining:     Optional[float] = None
    active_moisture:     bool  = False


@dataclass
class SafeTreatment:
    """One validated intervention the practitioner may apply."""
    treatment_id: str
    name:         str
    type:         str
    application:  str
    source:       str
    rank:         int = 1
    notes:        str = ""


@dataclass
class DangerousTreatment:
    """One intervention that must NOT be applied — with harm explanation."""
    treatment_id: str
    name:         str
    risk:         str
    mechanism:    str
    source:       str


@dataclass
class TreatmentProtocol:
    """
    Complete field treatment protocol for one artefact.

    This is the object the dashboard renders as green/red protocol panel.
    """
    artefact_id:       str
    substrate_id:      str
    substrate_name:    str
    decay_processes:   list[str] = field(default_factory=list)
    decay_names:       list[str] = field(default_factory=list)
    safe_treatments:   list[SafeTreatment] = field(default_factory=list)
    unsafe_treatments: list[DangerousTreatment] = field(default_factory=list)
    treatment_sequence: str = ""
    urgency:           str = "MONITOR"
    urgency_level:     int = 1
    urgency_color:     str = "#27ae60"
    urgency_action:    str = ""
    overall_risk_score: float = 0.0
    sources:           list[str] = field(default_factory=list)
    confidence_notes:  list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise for WebSocket / JSON catalogue export."""
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — KNOWLEDGE BASE LOADER
# ═══════════════════════════════════════════════════════════════

class ConservationKB:
    """
    Loads and indexes the conservation knowledge base JSON.

    The KB is read once at init and kept in memory. At ~1–5 MB this is
    negligible on the laptop compute node.
    """

    def __init__(self, kb_path: Path = CONSERVATION_KB_PATH):
        """
        Load conservation_kb.json from disk.

        Args:
            kb_path: Path to merged KB JSON (built by build_conservation_kb.py)
        """
        self.path = Path(kb_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Conservation KB not found at {self.path}. "
                "Run: python scripts/build_conservation_kb.py"
            )
        with open(self.path, encoding="utf-8") as f:
            self.data = json.load(f)

        self.substrates   = self.data.get("substrates", {})
        self.decay        = self.data.get("decay_patterns", {})
        self.treatments   = self.data.get("treatments", {})
        self.matrix       = self.data.get("compatibility_matrix", {})
        self.urgency      = self.data.get("urgency_levels", {})
        self.artefact_map = self.data.get("artefact_substrate_map", ARTEFACT_SUBSTRATE_MAP)
        self.metadata     = self.data.get("metadata", {})

        logger.info(
            f"Conservation KB loaded: {len(self.substrates)} substrates, "
            f"{len(self.decay)} decay patterns, {len(self.treatments)} treatments"
        )


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DECAY INFERENCE FROM SENSOR INPUTS
# ═══════════════════════════════════════════════════════════════

def _infer_decay_processes(
    inputs: DiagnosticInputs,
    substrate_id: str,
    kb: ConservationKB,
) -> list[str]:
    """
    Map quantitative sensor readings to ICOMOS decay pattern IDs.

    Uses threshold rules aligned with existing AXUM detector outputs.
    Multiple decay processes can coexist on one artefact.

    Args:
        inputs:       Diagnostic sensor bundle
        substrate_id: Resolved substrate from KB
        kb:           Loaded knowledge base

    Returns:
        List of decay pattern IDs, ordered by severity relevance
    """
    detected: list[str] = []
    substrate = kb.substrates.get(substrate_id, {})
    common    = set(substrate.get("common_decay", []))

    # Salt crystallisation
    if inputs.salt_risk >= 0.15 or inputs.salt_critical:
        detected.append("salt_crystallization")
        if inputs.salt_critical:
            detected.append("subflorescence")

    # Mechanical cracking
    if inputs.crack_severity >= 0.25:
        detected.append("crack_mechanical")

    # Biological colonisation
    if inputs.biological_detected:
        detected.append("biological_colonization")

    # Inscription-specific biological crust (low OCR + inscription present)
    if inputs.has_inscription and inputs.ocr_confidence < 0.55:
        detected.append("biological_crust_inscription")

    # Surface stress → granular disintegration / spalling risk
    if inputs.stress_score >= 0.45:
        detected.append("granular_disintegration")
    if inputs.stress_score >= 0.65:
        detected.append("spalling")

    # Low hardness → powdering / erosion
    if inputs.hardness_score is not None and inputs.hardness_score < 0.35:
        detected.append("granular_disintegration")

    # Organic substrates — mould from high moisture proxy
    if substrate_id in ("parchment_vellum", "wood", "textile_linen", "paper"):
        if inputs.active_moisture or inputs.biological_detected:
            detected.append("mould_biological_organic")
        if substrate_id == "parchment_vellum" and inputs.ocr_confidence < 0.5:
            detected.append("ink_corrosion")

    # Metals
    if substrate_id == "metal_bronze" and inputs.salt_risk >= 0.1:
        detected.append("bronze_disease")

    if substrate_id == "metal_iron_steel":
        detected.append("rust_corrosion")

    # Painted surfaces
    if substrate_id == "painted_surface_fresco" and inputs.stress_score >= 0.5:
        detected.append("paint_detachment")

    # Capillary rise on porous earthen/volcanic stones
    if substrate_id in ("tuff_ignimbrite", "earthen_mud_brick") and inputs.active_moisture:
        detected.append("capillary_rise_damage")

    # Fallback: at least one decay from substrate's common patterns
    if not detected and common:
        detected.append(next(iter(common)))

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in detected:
        if d not in seen and d in kb.decay:
            seen.add(d)
            unique.append(d)

    return unique


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — TREATMENT COMPATIBILITY ENGINE
# ═══════════════════════════════════════════════════════════════

class TreatmentAdvisor:
    """
    Core decision engine: sensor inputs → TreatmentProtocol.

    Queries the compatibility matrix for each substrate+decay pair,
    merges results, ranks safe treatments, and escalates urgency.
    """

    def __init__(self, kb: Optional[ConservationKB] = None):
        """
        Initialise advisor with knowledge base.

        Args:
            kb: Pre-loaded KB instance, or None to load from config path
        """
        self.kb = kb or ConservationKB()

    def resolve_substrate(self, inputs: DiagnosticInputs) -> str:
        """
        Determine substrate ID from classifier label or override.

        Args:
            inputs: Diagnostic sensor bundle

        Returns:
            Substrate ID string present in conservation KB
        """
        if inputs.substrate_override and inputs.substrate_override in self.kb.substrates:
            return inputs.substrate_override

        mapped = self.kb.artefact_map.get(
            inputs.artefact_class,
            ARTEFACT_SUBSTRATE_MAP.get(inputs.artefact_class, "limestone_porous"),
        )
        if mapped not in self.kb.substrates:
            logger.warning(f"Substrate {mapped} not in KB — falling back to limestone_porous")
            return "limestone_porous"
        return mapped

    def _lookup_matrix(
        self,
        substrate_id: str,
        decay_id: str,
    ) -> dict:
        """
        Fetch compatibility entry for substrate+decay pair.

        Falls back to rule-based inference from treatment metadata when
        no explicit matrix entry exists.

        Args:
            substrate_id: KB substrate ID
            decay_id:       KB decay pattern ID

        Returns:
            Dict with safe, unsafe, sequence, safe_after_desalination keys
        """
        explicit = (
            self.kb.matrix
            .get(substrate_id, {})
            .get(decay_id, {})
        )
        if explicit:
            return explicit

        # Rule-based fallback from treatment definitions
        safe_ids: list[str] = []
        unsafe: list[dict] = []

        for tid, t in self.kb.treatments.items():
            ttype = t.get("type", "")
            compat_sub = t.get("compatible_substrates", [])
            compat_decay = t.get("compatible_decay", [])
            never_on = t.get("never_use_on", [])
            incompat_sub = t.get("incompatible_substrates", [])
            incompat_decay = t.get("incompatible_with_decay", [])

            if substrate_id in never_on or substrate_id in incompat_sub:
                unsafe.append({
                    "treatment": tid,
                    "risk": t.get("danger_level", "high"),
                    "mechanism": t.get("danger_reason", t.get("incompatibility_reason", "")),
                    "source": t.get("source", ""),
                })
                continue

            if decay_id in incompat_decay:
                unsafe.append({
                    "treatment": tid,
                    "risk": "high",
                    "mechanism": t.get("incompatibility_reason_decay", ""),
                    "source": t.get("source", ""),
                })
                continue

            if ttype == "DANGEROUS_TREATMENT":
                if substrate_id in never_on or not never_on:
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

        return {
            "safe": safe_ids,
            "unsafe": unsafe,
            "sequence": "",
            "safe_after_desalination": [],
        }

    def _build_safe_treatment(self, tid: str, rank: int) -> Optional[SafeTreatment]:
        """Convert treatment ID to SafeTreatment dataclass."""
        t = self.kb.treatments.get(tid)
        if not t or t.get("type") == "DANGEROUS_TREATMENT":
            return None
        return SafeTreatment(
            treatment_id=tid,
            name=t.get("name", tid),
            type=t.get("type", "unknown"),
            application=t.get("application", ""),
            source=t.get("source", ""),
            rank=rank,
            notes=t.get("notes", ""),
        )

    def _build_unsafe_treatment(self, entry: dict) -> DangerousTreatment:
        """Convert matrix unsafe entry to DangerousTreatment dataclass."""
        tid = entry.get("treatment", "unknown")
        t   = self.kb.treatments.get(tid, {})
        return DangerousTreatment(
            treatment_id=tid,
            name=t.get("name", tid),
            risk=entry.get("risk", "high"),
            mechanism=entry.get("mechanism", t.get("danger_reason", "")),
            source=entry.get("source", t.get("source", "")),
        )

    def _compute_urgency(
        self,
        inputs: DiagnosticInputs,
        decay_ids: list[str],
    ) -> tuple[str, int, str, str]:
        """
        Determine urgency level from sensor severity and decay types.

        Returns:
            (label, level, color, action) tuple
        """
        # Emergency triggers
        if inputs.salt_critical and inputs.has_inscription:
            return self._urgency_tuple("emergency")
        if inputs.crack_severity >= 0.85 and inputs.has_inscription:
            return self._urgency_tuple("emergency")
        if inputs.years_remaining is not None and inputs.years_remaining < TREATMENT_URGENCY_CRITICAL_YEARS:
            return self._urgency_tuple("emergency")

        # Critical triggers
        if inputs.salt_critical or inputs.salt_risk >= 0.6:
            return self._urgency_tuple("critical")
        if "biological_crust_inscription" in decay_ids:
            return self._urgency_tuple("critical")
        if inputs.crack_severity >= 0.7:
            return self._urgency_tuple("critical")
        if inputs.years_remaining is not None and inputs.years_remaining < TREATMENT_URGENCY_PRIORITY_YEARS:
            return self._urgency_tuple("critical")

        # Priority triggers
        if inputs.crack_severity >= 0.4 or inputs.salt_risk >= 0.3:
            return self._urgency_tuple("priority")
        if inputs.stress_score >= 0.5:
            return self._urgency_tuple("priority")

        # Monitor
        if decay_ids:
            return self._urgency_tuple("monitor")

        return self._urgency_tuple("stable")

    def _urgency_tuple(self, key: str) -> tuple[str, int, str, str]:
        """Look up urgency level metadata from KB."""
        u = self.kb.urgency.get(key, {})
        return (
            u.get("label", key.upper()),
            u.get("level", 1),
            u.get("color", "#27ae60"),
            u.get("action", ""),
        )

    def advise(self, inputs: DiagnosticInputs) -> TreatmentProtocol:
        """
        Main entry point: produce complete treatment protocol.

        Args:
            inputs: Sensor diagnostic bundle from pipeline

        Returns:
            TreatmentProtocol ready for dashboard / catalogue export
        """
        substrate_id = self.resolve_substrate(inputs)
        substrate    = self.kb.substrates.get(substrate_id, {})
        decay_ids    = _infer_decay_processes(inputs, substrate_id, self.kb)

        safe_map: dict[str, SafeTreatment] = {}
        unsafe_map: dict[str, DangerousTreatment] = {}
        sequences: list[str] = []
        rank = 1

        for decay_id in decay_ids:
            entry = self._lookup_matrix(substrate_id, decay_id)

            for tid in entry.get("safe", []):
                st = self._build_safe_treatment(tid, rank)
                if st and tid not in safe_map:
                    safe_map[tid] = st
                    rank += 1

            # Post-desalination treatments only when salt is active
            if decay_id in ("salt_crystallization", "subflorescence", "efflorescence"):
                if inputs.salt_risk < 0.4:
                    for tid in entry.get("safe_after_desalination", []):
                        st = self._build_safe_treatment(tid, rank)
                        if st and tid not in safe_map:
                            st.notes = "Apply only after desalination cycles complete"
                            safe_map[tid] = st
                            rank += 1

            for u in entry.get("unsafe", []):
                tid = u.get("treatment", "")
                if tid and tid not in unsafe_map:
                    unsafe_map[tid] = self._build_unsafe_treatment(u)

            seq = entry.get("sequence", "")
            if seq:
                sequences.append(seq)

        # Global dangerous treatments for this substrate
        for tid, t in self.kb.treatments.items():
            if t.get("type") != "DANGEROUS_TREATMENT":
                continue
            never = t.get("never_use_on", [])
            if substrate_id in never or not never:
                if tid not in unsafe_map:
                    unsafe_map[tid] = DangerousTreatment(
                        treatment_id=tid,
                        name=t.get("name", tid),
                        risk=t.get("danger_level", "critical"),
                        mechanism=t.get("danger_reason", ""),
                        source=t.get("source", ""),
                    )

        urgency_label, urgency_level, urgency_color, urgency_action = (
            self._compute_urgency(inputs, decay_ids)
        )

        # Overall risk score 0–1
        risk = min(1.0, (
            inputs.crack_severity * 0.35
            + inputs.salt_risk * 0.30
            + inputs.stress_score * 0.20
            + (1.0 - inputs.ocr_confidence) * 0.15
        ))

        decay_names = [
            self.kb.decay.get(d, {}).get("name", d) for d in decay_ids
        ]

        confidence_notes: list[str] = []
        if inputs.ocr_confidence < 0.5:
            confidence_notes.append(
                "Low OCR confidence — biological crust on inscription suspected"
            )
        if inputs.salt_critical:
            confidence_notes.append(
                "Active salt migration — desalination required before consolidants"
            )
        if not decay_ids:
            confidence_notes.append(
                "No decay patterns inferred — protocol based on substrate defaults"
            )

        protocol = TreatmentProtocol(
            artefact_id=inputs.artefact_id,
            substrate_id=substrate_id,
            substrate_name=substrate.get("name", substrate_id),
            decay_processes=decay_ids,
            decay_names=decay_names,
            safe_treatments=sorted(safe_map.values(), key=lambda x: x.rank),
            unsafe_treatments=list(unsafe_map.values()),
            treatment_sequence=" -> ".join(sequences) if sequences else "",
            urgency=urgency_label,
            urgency_level=urgency_level,
            urgency_color=urgency_color,
            urgency_action=urgency_action,
            overall_risk_score=round(risk, 3),
            sources=self.kb.metadata.get("sources", [])[:5],
            confidence_notes=confidence_notes,
        )

        logger.info(
            f"Treatment protocol for {inputs.artefact_id}: "
            f"{len(protocol.safe_treatments)} safe, "
            f"{len(protocol.unsafe_treatments)} unsafe, "
            f"urgency={protocol.urgency}"
        )
        return protocol


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_treatment_advisor(inputs: DiagnosticInputs) -> TreatmentProtocol:
    """
    Convenience wrapper used by main_pipeline and dashboard demo.

    Args:
        inputs: Diagnostic sensor bundle

    Returns:
        TreatmentProtocol for the artefact
    """
    advisor = TreatmentAdvisor()
    return advisor.advise(inputs)


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — STANDALONE SYNTHETIC TEST
# ═══════════════════════════════════════════════════════════════

def _synthetic_test() -> None:
    """
    Run advisor on synthetic limestone inscription case WITHOUT hardware.

    Simulates: porous limestone + active salt + hairline crack + bio crust.
    Expected: desalination safe, acrylic sealant unsafe, urgency CRITICAL+.
    """
    logger.info("=== Treatment Advisor Synthetic Test ===")

    inputs = DiagnosticInputs(
        artefact_id="AX002",
        artefact_class="inscription_fragment",
        crack_severity=0.55,
        salt_risk=0.72,
        salt_critical=True,
        stress_score=0.38,
        ocr_confidence=0.41,
        has_inscription=True,
        hardness_score=0.45,
        biological_detected=True,
        years_remaining=12.0,
        active_moisture=True,
    )

    protocol = run_treatment_advisor(inputs)

    print(f"\nTREATMENT PROTOCOL - {protocol.artefact_id}")
    print("=" * 50)
    print(f"Substrate:  {protocol.substrate_name}")
    print(f"Decay:      {', '.join(protocol.decay_names)}")
    print(f"Urgency:    {protocol.urgency} - {protocol.urgency_action}")
    print(f"Risk score: {protocol.overall_risk_score:.2f}")
    print()

    print("[SAFE] TO APPLY:")
    for i, t in enumerate(protocol.safe_treatments[:5], 1):
        print(f"  {i}. {t.name}")
        if t.application:
            print(f"     {t.application[:80]}...")
    print()

    print("[UNSAFE] DO NOT APPLY:")
    for i, t in enumerate(protocol.unsafe_treatments[:5], 1):
        print(f"  {i}. {t.name} [{t.risk}]")
        print(f"     {t.mechanism[:80]}...")
    print()

    if protocol.treatment_sequence:
        seq = protocol.treatment_sequence.encode("ascii", "replace").decode("ascii")
        print(f"Sequence: {seq[:120]}...")
    print()

    # Assertions for CI-less self-check
    unsafe_names = {t.name.lower() for t in protocol.unsafe_treatments}
    assert any("acrylic" in n for n in unsafe_names), "Acrylic sealant should be flagged unsafe"
    assert protocol.urgency in ("EMERGENCY", "CRITICAL"), f"Expected high urgency, got {protocol.urgency}"
    assert len(protocol.safe_treatments) > 0, "Should recommend at least one safe treatment"

    logger.success("Synthetic test passed")


if __name__ == "__main__":
    _synthetic_test()
