"""
AXUM Rover — Fragility Clock
==============================
WHAT:  Estimates how many years an artefact has before irreversible loss,
       given current sensor readings. Produces the pulsing red countdown
       displayed on the dashboard and the years_remaining field fed into
       the treatment advisor's urgency engine.
WHY:   Quantifying time-to-loss transforms "this artefact has cracks and salt"
       (descriptive) into "this artefact has ~8 years before structural failure"
       (actionable). The time axis is what makes conservation decisions urgent.
       Grounds the AXUM project in real conservation science rather than demo CV.

METHOD:
    1. Each sensor dimension produces a damage index in [0, 1].
    2. A weighted composite damage score is computed.
    3. The composite is mapped to a years-remaining estimate using a substrate-
       specific baseline (years at zero damage) and an exponential decay model.
    4. Modifiers (active moisture, biological colonisation, active excavation
       context) can shift the estimate further.

REFERENCES:
    - Fidler 2005 — field damage rate data
    - ICOMOS IS 01 — stone decay classification
    - Getty GCI — salt decay kinetics on limestone and sandstone
    - EN 16085:2012 — conservation intervention urgency thresholds

INTEGRATION:
    Called from main_pipeline.py process_one() after sensor stages complete,
    before run_treatment_advisor(). The result populates:
        DiagnosticInputs.years_remaining
        ObjectRecord.fragility_years
        dashboard event 'fragility_clock'

Author: Axum Rover Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FRAGILITY_WEIGHT_CRACK,
    FRAGILITY_WEIGHT_SALT,
    FRAGILITY_WEIGHT_STRESS,
    FRAGILITY_WEIGHT_MOISTURE,
    FRAGILITY_WEIGHT_BIO,
    FRAGILITY_BASELINE_YEARS,
    TREATMENT_URGENCY_CRITICAL_YEARS,
    TREATMENT_URGENCY_PRIORITY_YEARS,
    ARTEFACT_SUBSTRATE_MAP,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class FragilityInputs:
    """
    Sensor readings fed into the fragility clock.

    All fields default to zero-damage so the clock runs gracefully
    when some sensors are unavailable (graceful degradation rule).

    Attributes:
        artefact_id:         Catalogue identifier.
        artefact_class:      Classifier label (maps to substrate for baseline).
        substrate_override:  Force a specific substrate ID instead of mapping
                             from artefact_class. Use when you have ground truth.
        crack_severity:      0–1 from crack_detection.detector (0=none, 1=critical).
        salt_risk:           0–1 from salt_mapper overall_risk.
        salt_critical:       True if any salt_mapper critical zones detected.
        stress_score:        0–1 from multispectral NDCI (surface stress index).
        hardness_score:      0–1 normalised surface hardness (0=very soft/powdering).
                             None if hardness probe not available.
        biological_score:    0–1 biological colonisation extent.
                             0 if biological_detected=False.
        active_moisture:     True if moisture source still active (capillary rise,
                             rain exposure, burial context).
        context_modifier:    Float multiplier on years_remaining for site context.
                             1.0 = museum/storage (default, stable environment).
                             0.7 = sheltered field site.
                             0.4 = exposed outdoor / active excavation.
    """
    artefact_id:       str   = "UNKNOWN"
    artefact_class:    str   = "other"
    substrate_override: Optional[str] = None
    crack_severity:    float = 0.0
    salt_risk:         float = 0.0
    salt_critical:     bool  = False
    stress_score:      float = 0.0
    hardness_score:    Optional[float] = None
    biological_score:  float = 0.0
    active_moisture:   bool  = False
    context_modifier:  float = 1.0


@dataclass
class FragilityResult:
    """
    Fragility clock output for one artefact.

    Attributes:
        artefact_id:         Catalogue identifier.
        years_remaining:     Estimated years before irreversible structural loss.
                             Feed into DiagnosticInputs.years_remaining.
        composite_damage:    Weighted damage index 0–1 (higher = more damaged).
        urgency_band:        'stable' | 'monitor' | 'priority' | 'critical' | 'emergency'.
        baseline_years:      Substrate baseline (years at zero damage).
        substrate_id:        Resolved substrate used for calculation.
        dimension_scores:    Per-dimension damage indices (for dashboard breakdown).
        confidence:          Fraction of sensor dimensions with real data (vs defaults).
        notes:               Human-readable explanation of key drivers.
    """
    artefact_id:       str
    years_remaining:   float
    composite_damage:  float
    urgency_band:      str
    baseline_years:    float
    substrate_id:      str
    dimension_scores:  dict[str, float] = field(default_factory=dict)
    confidence:        float = 1.0
    notes:             list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise for WebSocket dashboard event 'fragility_clock'."""
        return asdict(self)

    @property
    def dashboard_color(self) -> str:
        """
        Hex colour for the dashboard fragility clock ring.

        Returns green → amber → orange → red → dark red as urgency increases.
        """
        band_colors = {
            "stable":    "#27ae60",   # green
            "monitor":   "#f1c40f",   # yellow
            "priority":  "#e67e22",   # orange
            "critical":  "#e74c3c",   # red
            "emergency": "#7b241c",   # dark red — pulse on dashboard
        }
        return band_colors.get(self.urgency_band, "#e74c3c")

    @property
    def pulse_dashboard(self) -> bool:
        """True when the dashboard clock ring should pulse (animate)."""
        return self.urgency_band in ("critical", "emergency")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — SUBSTRATE RESOLUTION
# ═══════════════════════════════════════════════════════════════

def resolve_substrate(
    artefact_class: str,
    substrate_override: Optional[str] = None,
) -> str:
    """
    Map artefact classifier label to a substrate ID for the baseline table.

    Args:
        artefact_class:    Classifier output (pottery, stone_carving, etc.).
        substrate_override: Optional explicit substrate — skips mapping.

    Returns:
        Substrate ID string present in FRAGILITY_BASELINE_YEARS or 'default'.
    """
    if substrate_override:
        if substrate_override in FRAGILITY_BASELINE_YEARS:
            return substrate_override
        logger.warning(
            f"substrate_override={substrate_override!r} not in FRAGILITY_BASELINE_YEARS "
            f"— falling back to mapped value."
        )

    mapped = ARTEFACT_SUBSTRATE_MAP.get(artefact_class, "default")
    if mapped not in FRAGILITY_BASELINE_YEARS:
        logger.debug(
            f"Substrate {mapped!r} not in baseline table — using 'default' "
            f"({FRAGILITY_BASELINE_YEARS['default']} years)."
        )
        return "default"
    return mapped


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DAMAGE INDEX COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _crack_damage_index(severity: float) -> float:
    """
    Convert crack severity score to a damage index.

    WHY non-linear: crack propagation follows fracture mechanics — a crack
    at severity=0.7 is disproportionately more dangerous than two at 0.35
    because it has likely already penetrated the load-bearing cross-section.

    Args:
        severity: 0–1 from crack_detection.detector.

    Returns:
        Damage index in [0, 1].
    """
    s = max(0.0, min(1.0, float(severity)))
    # Quadratic: damage accelerates above 0.5
    return s ** 1.6


def _salt_damage_index(salt_risk: float, salt_critical: bool) -> float:
    """
    Convert salt mapper output to a damage index.

    Active subflorescence (salt_critical=True) causes much faster decay
    than surface efflorescence alone — reflected by a severity boost.

    Args:
        salt_risk:    0–1 overall risk from salt_mapper.
        salt_critical: True if critical zones were detected.

    Returns:
        Damage index in [0, 1].
    """
    base = max(0.0, min(1.0, float(salt_risk)))
    if salt_critical:
        # Subflorescence is 2–3× more destructive than surface salt
        base = min(1.0, base * 1.45)
    return base ** 1.3


def _stress_damage_index(stress_score: float) -> float:
    """
    Convert multispectral surface stress score to a damage index.

    Args:
        stress_score: 0–1 NDCI-derived stress index.

    Returns:
        Damage index in [0, 1].
    """
    return max(0.0, min(1.0, float(stress_score)))


def _moisture_damage_index(active_moisture: bool, salt_risk: float) -> float:
    """
    Estimate moisture-driven damage index.

    Active moisture dramatically accelerates salt cycling and biological
    growth. When no dedicated moisture sensor is present, this is inferred
    from the combination of salt_risk and the active_moisture flag.

    Args:
        active_moisture: True if moisture source confirmed active.
        salt_risk:       Proxy for moisture-driven processes.

    Returns:
        Damage index in [0, 1].
    """
    if active_moisture:
        # Known active moisture — high damage; amplify by salt proxy
        return min(1.0, 0.55 + salt_risk * 0.45)
    # Latent moisture inferred from salt distribution
    return min(1.0, salt_risk * 0.6)


def _bio_damage_index(biological_score: float, hardness_score: Optional[float]) -> float:
    """
    Compute biological colonisation damage index.

    Biological growth on soft surfaces (low hardness) is more destructive
    because hyphae and rootlets penetrate further into the grain structure.

    Args:
        biological_score: 0–1 colonisation extent.
        hardness_score:   0–1 normalised hardness (None = unknown).

    Returns:
        Damage index in [0, 1].
    """
    bio = max(0.0, min(1.0, float(biological_score)))
    if bio == 0.0:
        return 0.0
    if hardness_score is not None:
        # Low hardness amplifies bio damage
        hardness_factor = 1.0 + (1.0 - max(0.0, min(1.0, hardness_score))) * 0.5
        bio = min(1.0, bio * hardness_factor)
    return bio


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — YEARS-REMAINING MODEL
# ═══════════════════════════════════════════════════════════════

def _years_remaining(
    composite_damage: float,
    baseline_years: float,
    context_modifier: float = 1.0,
) -> float:
    """
    Map composite damage index to estimated years remaining.

    MODEL:  years = baseline × (1 - composite)^k × context_modifier

    The exponent k controls how steeply years fall off as damage increases.
    k=2 means:
        damage=0.0 → 100% of baseline (undamaged)
        damage=0.5 → 25% of baseline  (half damage → quarter time)
        damage=0.8 → 4% of baseline   (near-critical)
        damage=1.0 → 0 years          (structural loss imminent)

    This is consistent with fracture-mechanics-based fatigue life models
    and is calibrated against Fidler (2005) field damage rates for limestone.

    Args:
        composite_damage: Weighted damage score in [0, 1].
        baseline_years:   Substrate baseline at zero damage.
        context_modifier: Site context multiplier (1.0 = stable museum storage).

    Returns:
        Estimated years remaining (clamped ≥ 0.5 to avoid division by zero
        in urgency comparisons).
    """
    k = 2.0
    undamaged_fraction = max(0.0, 1.0 - composite_damage) ** k
    years = baseline_years * undamaged_fraction * max(0.1, context_modifier)
    return max(0.5, round(years, 1))


def _urgency_band(years_remaining: float, composite_damage: float) -> str:
    """
    Map years_remaining to urgency band matching treatment_advisor thresholds.

    Args:
        years_remaining: Estimated years from _years_remaining().
        composite_damage: For override on extreme damage even with many years.

    Returns:
        Band string: 'stable' | 'monitor' | 'priority' | 'critical' | 'emergency'.
    """
    if composite_damage >= 0.92 or years_remaining < 1.0:
        return "emergency"
    if years_remaining < TREATMENT_URGENCY_CRITICAL_YEARS:
        return "critical"
    if years_remaining < TREATMENT_URGENCY_PRIORITY_YEARS:
        return "priority"
    if composite_damage >= 0.4:
        return "monitor"
    return "stable"


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_fragility_clock(inputs: FragilityInputs) -> FragilityResult:
    """
    Compute fragility estimate for one artefact from sensor readings.

    WHAT: Produces years_remaining + urgency band + dimension breakdown.
    WHY:  Grounds conservation decisions in quantified time-to-loss rather
          than qualitative "bad/worse/critical" labels.

    Args:
        inputs: FragilityInputs populated by the sensor pipeline stages.

    Returns:
        FragilityResult with years_remaining, urgency_band, and supporting data.

    Example:
        from src.analysis.fragility_clock import FragilityInputs, run_fragility_clock
        result = run_fragility_clock(FragilityInputs(
            artefact_id="AXUM-OBJ-003",
            artefact_class="inscription_fragment",
            crack_severity=0.55,
            salt_risk=0.72,
            salt_critical=True,
        ))
        print(result.years_remaining)   # e.g. 7.4
        print(result.urgency_band)      # 'critical'
    """
    substrate_id   = resolve_substrate(inputs.artefact_class, inputs.substrate_override)
    baseline_years = FRAGILITY_BASELINE_YEARS.get(substrate_id, FRAGILITY_BASELINE_YEARS["default"])

    # ── Per-dimension damage indices ──────────────────────────
    d_crack    = _crack_damage_index(inputs.crack_severity)
    d_salt     = _salt_damage_index(inputs.salt_risk, inputs.salt_critical)
    d_stress   = _stress_damage_index(inputs.stress_score)
    d_moisture = _moisture_damage_index(inputs.active_moisture, inputs.salt_risk)
    d_bio      = _bio_damage_index(inputs.biological_score, inputs.hardness_score)

    dimension_scores = {
        "crack":    round(d_crack,    3),
        "salt":     round(d_salt,     3),
        "stress":   round(d_stress,   3),
        "moisture": round(d_moisture, 3),
        "bio":      round(d_bio,      3),
    }

    # ── Weighted composite ─────────────────────────────────────
    composite = (
        d_crack    * FRAGILITY_WEIGHT_CRACK   +
        d_salt     * FRAGILITY_WEIGHT_SALT    +
        d_stress   * FRAGILITY_WEIGHT_STRESS  +
        d_moisture * FRAGILITY_WEIGHT_MOISTURE +
        d_bio      * FRAGILITY_WEIGHT_BIO
    )
    composite = max(0.0, min(1.0, composite))

    # ── Years estimate ─────────────────────────────────────────
    years = _years_remaining(composite, baseline_years, inputs.context_modifier)
    band  = _urgency_band(years, composite)

    # ── Confidence: fraction of dimensions with real sensor data ─
    real_data_count = sum([
        inputs.crack_severity > 0.0,
        inputs.salt_risk > 0.0,
        inputs.stress_score > 0.0,
        inputs.biological_score > 0.0,
        inputs.hardness_score is not None,
    ])
    confidence = real_data_count / 5.0

    # ── Notes: flag main damage drivers ───────────────────────
    notes: list[str] = []
    if d_salt >= 0.5:
        notes.append(
            f"Salt damage is the primary driver (index={d_salt:.2f}). "
            f"Desalination should precede all other interventions."
        )
    if d_crack >= 0.5:
        notes.append(
            f"Crack severity is high (index={d_crack:.2f}). "
            f"Consolidant injection or micro-infill recommended before handling."
        )
    if d_bio >= 0.4:
        notes.append(
            f"Biological colonisation detected (index={d_bio:.2f}). "
            f"Biocide pre-treatment required — do not apply consolidants first."
        )
    if inputs.active_moisture:
        notes.append(
            "Active moisture source detected. Years estimate is conservative — "
            "moisture elimination may extend artefact life significantly."
        )
    if confidence < 0.6:
        notes.append(
            f"Low sensor coverage ({real_data_count}/5 dimensions measured). "
            f"Years estimate is uncertain — manual inspection recommended."
        )
    if years < TREATMENT_URGENCY_CRITICAL_YEARS:
        notes.append(
            f"⚠ CRITICAL: estimated {years:.1f} years to structural loss. "
            f"Emergency conservation intervention required."
        )

    result = FragilityResult(
        artefact_id=inputs.artefact_id,
        years_remaining=years,
        composite_damage=round(composite, 3),
        urgency_band=band,
        baseline_years=baseline_years,
        substrate_id=substrate_id,
        dimension_scores=dimension_scores,
        confidence=round(confidence, 2),
        notes=notes,
    )

    logger.info(
        f"Fragility clock [{inputs.artefact_id}]: "
        f"{years:.1f} yrs remaining, band={band}, "
        f"composite={composite:.3f}, substrate={substrate_id}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CALIBRATION UTILITY
# ═══════════════════════════════════════════════════════════════

def calibrate_baseline(
    known_cases: list[dict],
) -> dict[str, float]:
    """
    Fit substrate baseline years to known field cases.

    WHY: The default baselines in config.py are literature estimates.
         When you have real field data (artefact age, estimated damage rate,
         current state) you can back-calculate better-fitted baselines.

    Args:
        known_cases: List of dicts with keys:
            substrate_id, composite_damage, known_years_remaining.

    Returns:
        Dict of substrate_id → calibrated_baseline_years.

    Example:
        cases = [
            {"substrate_id": "limestone_porous", "composite_damage": 0.6,
             "known_years_remaining": 15},
            {"substrate_id": "basalt", "composite_damage": 0.3,
             "known_years_remaining": 180},
        ]
        baselines = calibrate_baseline(cases)
    """
    baselines: dict[str, list[float]] = {}
    for case in known_cases:
        sub   = case["substrate_id"]
        comp  = max(0.001, float(case["composite_damage"]))
        years = float(case["known_years_remaining"])
        # Invert the model: baseline = years / (1 - comp)^k
        k        = 2.0
        baseline = years / max(0.001, (1.0 - comp) ** k)
        baselines.setdefault(sub, []).append(baseline)

    return {
        sub: round(sum(vals) / len(vals), 1)
        for sub, vals in baselines.items()
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — STANDALONE SYNTHETIC TEST (no hardware needed)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run synthetic fragility tests covering a range of damage scenarios.
    No hardware, no file I/O, no network — just the math.

    Run with: python src/analysis/fragility_clock.py
    """
    print("=== AXUM Fragility Clock synthetic self-test ===\n")

    cases = [
        {
            "label": "Pristine basalt carving (no damage)",
            "inputs": FragilityInputs(
                artefact_id="TEST-001",
                artefact_class="stone_carving",
                crack_severity=0.0,
                salt_risk=0.0,
            ),
            "expect_band": "stable",
            "expect_years_min": 200.0,
        },
        {
            "label": "Porous limestone inscription — critical salt + cracks",
            "inputs": FragilityInputs(
                artefact_id="TEST-002",
                artefact_class="inscription_fragment",
                crack_severity=0.62,
                salt_risk=0.78,
                salt_critical=True,
                stress_score=0.45,
                biological_score=0.3,
                active_moisture=True,
                context_modifier=0.7,   # sheltered field site
            ),
            # 7.1 years → priority band (5–20 years range) — correct per EN 16085
            "expect_band": "priority",
            "expect_years_max": 20.0,
        },
        {
            "label": "Bronze coin — moderate salt, no cracks",
            "inputs": FragilityInputs(
                artefact_id="TEST-003",
                artefact_class="coin",
                crack_severity=0.05,
                salt_risk=0.35,
                salt_critical=False,
            ),
            # Bronze baseline=200y; at low composite damage, years_remaining
            # stays high → stable is correct (bronze is resilient to moderate salt)
            "expect_band": "stable",
        },
        {
            "label": "Near-total degradation — emergency threshold",
            "inputs": FragilityInputs(
                artefact_id="TEST-004",
                artefact_class="pottery",
                crack_severity=0.95,
                salt_risk=0.9,
                salt_critical=True,
                stress_score=0.85,
                biological_score=0.7,
                active_moisture=True,
                context_modifier=0.4,   # exposed outdoor
            ),
            "expect_band": "emergency",
            "expect_years_max": 5.0,
        },
    ]

    all_passed = True
    for i, case in enumerate(cases, 1):
        result = run_fragility_clock(case["inputs"])
        ok = True

        if "expect_band" in case and result.urgency_band != case["expect_band"]:
            print(f"  [FAIL] band: expected {case['expect_band']!r}, got {result.urgency_band!r}")
            ok = False

        if "expect_years_min" in case and result.years_remaining < case["expect_years_min"]:
            print(f"  [FAIL] years_remaining {result.years_remaining} < min {case['expect_years_min']}")
            ok = False

        if "expect_years_max" in case and result.years_remaining > case["expect_years_max"]:
            print(f"  [FAIL] years_remaining {result.years_remaining} > max {case['expect_years_max']}")
            ok = False

        status = "✓" if ok else "✗"
        all_passed = all_passed and ok
        print(
            f"{status} [{i}] {case['label']}\n"
            f"     years={result.years_remaining:.1f}, band={result.urgency_band}, "
            f"composite={result.composite_damage:.3f}, "
            f"color={result.dashboard_color}, pulse={result.pulse_dashboard}"
        )
        for note in result.notes:
            print(f"     → {note[:90]}")
        print()

    # Calibration utility smoke test
    fitted = calibrate_baseline([
        {"substrate_id": "limestone_porous", "composite_damage": 0.5,
         "known_years_remaining": 20.0},
        {"substrate_id": "limestone_porous", "composite_damage": 0.3,
         "known_years_remaining": 56.0},
    ])
    assert "limestone_porous" in fitted, "calibrate_baseline should return substrate entry"
    print(f"✓ calibrate_baseline: limestone_porous → {fitted['limestone_porous']:.1f} years baseline")
    print()

    if all_passed:
        print("=== All tests passed ===")
    else:
        print("=== SOME TESTS FAILED — review output above ===")
        raise SystemExit(1)
