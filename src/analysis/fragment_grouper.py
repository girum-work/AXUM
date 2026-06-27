# src/analysis/fragment_grouper.py
"""
AXUM ROVER — Fragment Grouping Engine
======================================
Links scanned fragments that likely belong to the same original object.

Without grouping, three pottery shards appear as three unrelated catalogue
entries. Conservationists cannot see they are one vessel. This module runs
after each scan, comparing the new object against all prior scans using:

  1. Fracture surface geometry (Open3D ICP when available) — strongest
  2. Material consistency (density, colour histogram, UV signature)
  3. Object class match (pottery ↔ pottery)
  4. Inscription continuity (partial Ge'ez text alignment)

Groups are persisted to fragment_groups.json and written back onto each
ObjectRecord (group_id, group_conf, match_scores).

Author: Axum Rover Team
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FRAGMENT_GROUPS_JSON,
    FRAGMENT_MATCH_POSSIBLE_MIN,
    FRAGMENT_MATCH_CONFIRMED_MIN,
    FRAGMENT_ICP_MAX_DISTANCE_M,
    FRAGMENT_DENSITY_TOLERANCE,
)
from src.catalogue.records import ObjectRecord


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class FragmentGroup:
    """
    A set of fragments believed to belong to one original object.

    Attributes:
        group_id:               AXUM-GRP-001 style identifier
        confirmed_members:      Object IDs with match score > 85%
        possible_members:       Object IDs with match score 65–85%
        reconstruction_conf:    0–1 confidence that grouping is correct
        estimated_completeness: 0–1 rough fraction of original object present
        vessel_type:            Inferred object class / period label
        pairwise_scores:        "OBJ-A|OBJ-B" → match score for catalogue display
        notes:                  Human-readable grouping notes
    """
    group_id:               str
    confirmed_members:      list[str] = field(default_factory=list)
    possible_members:       list[str] = field(default_factory=list)
    reconstruction_conf:    float = 0.0
    estimated_completeness: float = 0.0
    vessel_type:            str = ""
    pairwise_scores:        dict[str, float] = field(default_factory=dict)
    notes:                  str = ""

    def all_members(self) -> list[str]:
        """Return every object ID in this group."""
        return list(dict.fromkeys(self.confirmed_members + self.possible_members))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FragmentGroup":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# Known Ge'ez word fragments for continuity heuristic
_GEEZ_KNOWN_SEQUENCES = [
    "ሰላም", "ዓጼ", "ማርያም", "ክርስቶስ", "ቅዱስ",
    "ነጉሥ", "አምላክ", "ጊዮርጊስ", "ኢትዮጵያ", "እግዚእ",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — MATCHING SIGNALS
# ═══════════════════════════════════════════════════════════════

def _pair_key(id_a: str, id_b: str) -> str:
    """Canonical key for pairwise score storage (sorted IDs)."""
    a, b = sorted([id_a, id_b])
    return f"{a}|{b}"


def _material_consistency_score(rec_a: ObjectRecord, rec_b: ObjectRecord) -> float:
    """
    Compare density, colour histogram, and UV signature.

    Large material differences veto a match even if geometry looks similar.

    Args:
        rec_a, rec_b: Two object records to compare

    Returns:
        0.0–1.0 material similarity score
    """
    scores: list[float] = []

    if rec_a.density_g_cm3 is not None and rec_b.density_g_cm3 is not None:
        diff = abs(rec_a.density_g_cm3 - rec_b.density_g_cm3)
        if diff > FRAGMENT_DENSITY_TOLERANCE:
            return 0.0
        density_score = max(0.0, 1.0 - diff / FRAGMENT_DENSITY_TOLERANCE)
        scores.append(density_score)

    if rec_a.colour_histogram and rec_b.colour_histogram:
        a = np.array(rec_a.colour_histogram, dtype=float)
        b = np.array(rec_b.colour_histogram, dtype=float)
        if a.shape == b.shape and a.sum() > 0 and b.sum() > 0:
            a /= a.sum()
            b /= b.sum()
            # Histogram intersection (1 = identical)
            intersection = float(np.minimum(a, b).sum())
            scores.append(intersection)

    if rec_a.uv_signature is not None and rec_b.uv_signature is not None:
        uv_diff = abs(rec_a.uv_signature - rec_b.uv_signature)
        scores.append(max(0.0, 1.0 - uv_diff))

    if not scores:
        return 0.5

    return float(np.mean(scores))


def _icp_fracture_match(mesh_path_a: str, mesh_path_b: str) -> float:
    """
    Compare 3D mesh geometry via Open3D ICP registration.

    Falls back to dimension-similarity heuristic when Open3D is unavailable
    or mesh files are missing (per project graceful-degradation rule).

    Args:
        mesh_path_a: Path to first .obj mesh
        mesh_path_b: Path to second .obj mesh

    Returns:
        0.0–1.0 geometric fit score
    """
    path_a = Path(mesh_path_a) if mesh_path_a else None
    path_b = Path(mesh_path_b) if mesh_path_b else None

    if not path_a or not path_b or not path_a.exists() or not path_b.exists():
        return 0.0

    try:
        import open3d as o3d

        mesh_a = o3d.io.read_triangle_mesh(str(path_a))
        mesh_b = o3d.io.read_triangle_mesh(str(path_b))

        if mesh_a.is_empty() or mesh_b.is_empty():
            return 0.0

        pcd_a = mesh_a.sample_points_uniformly(5000)
        pcd_b = mesh_b.sample_points_uniformly(5000)

        result = o3d.pipelines.registration.registration_icp(
            pcd_a, pcd_b,
            FRAGMENT_ICP_MAX_DISTANCE_M,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        fitness = result.fitness
        rmse = result.inlier_rmse
        score = fitness * max(0.0, 1.0 - rmse / FRAGMENT_ICP_MAX_DISTANCE_M)
        return float(np.clip(score, 0.0, 1.0))

    except ImportError:
        logger.debug("Open3D not installed — using dimension fallback for ICP")
        return _dimension_similarity_fallback(path_a, path_b)
    except Exception as e:
        logger.warning(f"ICP fracture match failed: {e}")
        return _dimension_similarity_fallback(path_a, path_b)


def _dimension_similarity_fallback(path_a: Path, path_b: Path) -> float:
    """
    Rough geometric proxy when Open3D ICP cannot run.

    Compares file size ratio as weak signal (same order of magnitude meshes
    from similar fragments). Real ICP requires Open3D + actual meshes.

    Args:
        path_a, path_b: Mesh file paths

    Returns:
        Low-confidence 0.0–0.5 score
    """
    try:
        size_a = path_a.stat().st_size
        size_b = path_b.stat().st_size
        if size_a == 0 or size_b == 0:
            return 0.0
        ratio = min(size_a, size_b) / max(size_a, size_b)
        return float(ratio * 0.4)
    except OSError:
        return 0.0


def _check_inscription_continuity(text_a: str, text_b: str) -> float:
    """
    Check whether two partial inscriptions could be one continuous text.

    Uses overlap heuristic on Ge'ez known words. Full LLM continuity check
    can be added later via llm_restoration.py.

    Args:
        text_a, text_b: OCR strings (may contain [MISSING] tokens)

    Returns:
        0.0–1.0 continuity score
    """
    clean_a = text_a.replace("[MISSING]", "").replace("◌", "").strip()
    clean_b = text_b.replace("[MISSING]", "").replace("◌", "").strip()

    if not clean_a or not clean_b:
        return 0.0

    combined = clean_a[-3:] + clean_b[:3]
    for known in _GEEZ_KNOWN_SEQUENCES:
        if known in combined or combined in known:
            return 0.85
        if clean_a[-2:] in known and known.startswith(clean_b[:2]):
            return 0.80
        if clean_b[:2] in known and known.endswith(clean_a[-2:]):
            return 0.80

    return 0.2


def compute_match_score(rec_a: ObjectRecord, rec_b: ObjectRecord) -> float:
    """
    Weighted combination of all fragment-matching signals.

    Weights follow AXUM spec: ICP 40%, material 25%, class 20%, inscription 15%.

    Args:
        rec_a, rec_b: Two scanned object records

    Returns:
        0.0–1.0 match probability
    """
    signals: list[tuple[str, float, float]] = []

    material = _material_consistency_score(rec_a, rec_b)
    if material == 0.0 and rec_a.density_g_cm3 and rec_b.density_g_cm3:
        return 0.0
    signals.append(("material", material, 0.25))

    class_match = 1.0 if rec_a.class_name == rec_b.class_name else 0.0
    signals.append(("class", class_match, 0.20))

    if rec_a.mesh_path and rec_b.mesh_path:
        icp_score = _icp_fracture_match(rec_a.mesh_path, rec_b.mesh_path)
        signals.append(("fracture_icp", icp_score, 0.40))

    if rec_a.inscription_text and rec_b.inscription_text:
        continuity = _check_inscription_continuity(
            rec_a.inscription_text, rec_b.inscription_text
        )
        signals.append(("inscription", continuity, 0.15))

    if not signals:
        return 0.0

    total_weight = sum(w for _, _, w in signals)
    weighted_sum = sum(s * w for _, s, w in signals)
    return weighted_sum / total_weight


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — GROUP MANAGER
# ═══════════════════════════════════════════════════════════════

class FragmentGrouper:
    """
    Running fragment-group database for one mission (or persistent registry).

    Call register_object() after each scan. Groups and pairwise scores are
    saved to FRAGMENT_GROUPS_JSON and reflected on each ObjectRecord.
    """

    def __init__(self, persist_path: Path = FRAGMENT_GROUPS_JSON):
        """
        Initialise empty grouper or load existing groups from disk.

        Args:
            persist_path: JSON file for group persistence
        """
        self.persist_path = Path(persist_path)
        self.groups: list[FragmentGroup] = []
        self.objects: dict[str, ObjectRecord] = {}
        self.matches: dict[tuple[str, str], float] = {}
        self._load()

    def _load(self) -> None:
        """Load persisted groups from JSON if file exists."""
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            self.groups = [FragmentGroup.from_dict(g) for g in data.get("groups", [])]
            for oid, rec_dict in data.get("objects", {}).items():
                self.objects[oid] = ObjectRecord.from_dict(rec_dict)
            for key, score in data.get("matches", {}).items():
                parts = key.split("|")
                if len(parts) == 2:
                    self.matches[(parts[0], parts[1])] = score
            logger.info(
                f"Fragment grouper loaded: {len(self.groups)} groups, "
                f"{len(self.objects)} objects"
            )
        except Exception as e:
            logger.warning(f"Could not load fragment groups: {e}")

    def save(self) -> None:
        """Persist groups, objects, and pairwise matches to JSON."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "groups": [g.to_dict() for g in self.groups],
            "objects": {oid: rec.to_dict() for oid, rec in self.objects.items()},
            "matches": {_pair_key(a, b): s for (a, b), s in self.matches.items()},
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Fragment groups saved: {self.persist_path}")

    def register_object(
        self,
        record: Union[ObjectRecord, dict],
    ) -> list[tuple[str, float]]:
        """
        Register a newly scanned object and check for fragment matches.

        Updates groups, writes group_id onto the record, and persists state.

        Args:
            record: ObjectRecord or dict from catalogue pipeline

        Returns:
            List of (other_object_id, match_score) where score > 0.65
        """
        if isinstance(record, dict):
            record = ObjectRecord.from_dict(record)

        obj_id = record.object_id
        self.objects[obj_id] = record

        new_matches: list[tuple[str, float]] = []

        for existing_id, existing in self.objects.items():
            if existing_id == obj_id:
                continue

            score = compute_match_score(record, existing)
            if score >= FRAGMENT_MATCH_POSSIBLE_MIN:
                key = (obj_id, existing_id)
                self.matches[key] = score
                new_matches.append((existing_id, score))
                logger.info(
                    f"Fragment match: {obj_id} <-> {existing_id} ({score:.0%})"
                )

        record.match_scores = {
            other_id: round(score, 3)
            for other_id, score in new_matches
        }

        if new_matches:
            self._update_groups(obj_id, new_matches)
        else:
            record.group_id = None
            record.group_role = "ungrouped"
            record.group_conf = 0.0

        self._apply_group_to_record(record)
        self.save()
        return new_matches

    def _apply_group_to_record(self, record: ObjectRecord) -> None:
        """Copy group metadata from FragmentGroup onto ObjectRecord."""
        group = self.get_group_for_object(record.object_id)
        if not group:
            return
        record.group_id = group.group_id
        record.group_conf = group.reconstruction_conf
        if record.object_id in group.confirmed_members:
            record.group_role = "confirmed"
        elif record.object_id in group.possible_members:
            record.group_role = "possible"

    def _update_groups(self, new_id: str, new_matches: list[tuple[str, float]]) -> None:
        """Add object to existing group or create a new FragmentGroup."""
        matched_ids = [m[0] for m in new_matches]

        for group in self.groups:
            members = group.all_members()
            for matched_id in matched_ids:
                if matched_id in members:
                    score = max(s for oid, s in new_matches if oid == matched_id)
                    if score >= FRAGMENT_MATCH_CONFIRMED_MIN:
                        if new_id not in group.confirmed_members:
                            group.confirmed_members.append(new_id)
                        if matched_id in group.possible_members:
                            group.possible_members.remove(matched_id)
                            group.confirmed_members.append(matched_id)
                    elif new_id not in group.all_members():
                        group.possible_members.append(new_id)

                    for oid, s in new_matches:
                        group.pairwise_scores[_pair_key(new_id, oid)] = round(s, 3)

                    self._update_group_stats(group)
                    self._sync_all_members(group)
                    return

        confirmed = [m[0] for m in new_matches if m[1] >= FRAGMENT_MATCH_CONFIRMED_MIN]
        possible = [
            m[0] for m in new_matches
            if FRAGMENT_MATCH_POSSIBLE_MIN <= m[1] < FRAGMENT_MATCH_CONFIRMED_MIN
        ]

        new_group = FragmentGroup(
            group_id=f"AXUM-GRP-{len(self.groups) + 1:03d}",
            confirmed_members=list(dict.fromkeys([new_id] + confirmed)),
            possible_members=possible,
        )
        for oid, s in new_matches:
            new_group.pairwise_scores[_pair_key(new_id, oid)] = round(s, 3)

        self._update_group_stats(new_group)
        self.groups.append(new_group)
        self._sync_all_members(new_group)

        logger.info(
            f"New fragment group: {new_group.group_id} "
            f"({len(new_group.confirmed_members)} confirmed, "
            f"{len(new_group.possible_members)} possible)"
        )

    def _sync_all_members(self, group: FragmentGroup) -> None:
        """Refresh group_id/group_conf on every member ObjectRecord."""
        for oid in group.all_members():
            if oid in self.objects:
                self._apply_group_to_record(self.objects[oid])

    def _update_group_stats(self, group: FragmentGroup) -> None:
        """Recalculate reconstruction confidence and completeness estimates."""
        n_confirmed = len(group.confirmed_members)
        n_possible = len(group.possible_members)

        group.reconstruction_conf = min(0.95, 0.5 + n_confirmed * 0.12)
        group.estimated_completeness = min(
            1.0, n_confirmed * 0.13 + n_possible * 0.05
        )

        if group.confirmed_members:
            first = self.objects.get(group.confirmed_members[0])
            if first:
                group.vessel_type = first.class_name.replace("_", " ")

    def get_group_for_object(self, object_id: str) -> Optional[FragmentGroup]:
        """Return the FragmentGroup containing object_id, if any."""
        for group in self.groups:
            if object_id in group.confirmed_members or \
               object_id in group.possible_members:
                return group
        return None

    def get_group(self, group_id: str) -> Optional[FragmentGroup]:
        """Look up group by AXUM-GRP-xxx ID."""
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None

    def summary_for_api(self) -> list[dict]:
        """Serialise all groups for Flask /api/fragment-groups endpoint."""
        result = []
        for group in self.groups:
            result.append({
                **group.to_dict(),
                "member_records": [
                    self.objects[oid].to_dict()
                    for oid in group.all_members()
                    if oid in self.objects
                ],
            })
        return result


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — MAIN ENTRY + SYNTHETIC TEST
# ═══════════════════════════════════════════════════════════════

def run_fragment_grouping(record: ObjectRecord) -> tuple[list[tuple[str, float]], FragmentGrouper]:
    """
    Convenience wrapper: register one object and return matches + grouper.

    Args:
        record: Newly scanned object

    Returns:
        (new_matches, grouper) tuple
    """
    grouper = FragmentGrouper()
    matches = grouper.register_object(record)
    return matches, grouper


def _synthetic_test() -> None:
    """
    Simulate three pottery shards grouping WITHOUT hardware or Open3D.

    Uses matching density/class/histogram to confirm grouping logic.
    """
    import tempfile

    logger.info("=== Fragment Grouper Synthetic Test ===")

    test_path = Path(tempfile.gettempdir()) / "axum_fragment_test_groups.json"
    grouper = FragmentGrouper(persist_path=test_path)

    hist_pottery = [0.1, 0.2, 0.15, 0.1, 0.05, 0.1, 0.1, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.05]

    shards = [
        ObjectRecord(
            object_id="AXUM-OBJ-001",
            sequence_number=1,
            class_name="pottery",
            class_confidence=0.91,
            density_g_cm3=1.84,
            colour_histogram=hist_pottery,
            uv_signature=0.32,
        ),
        ObjectRecord(
            object_id="AXUM-OBJ-002",
            sequence_number=2,
            class_name="pottery",
            class_confidence=0.88,
            density_g_cm3=1.79,
            colour_histogram=[h * 0.95 + 0.01 for h in hist_pottery],
            uv_signature=0.30,
            inscription_text="ሰ[MISSING]",
        ),
        ObjectRecord(
            object_id="AXUM-OBJ-003",
            sequence_number=3,
            class_name="pottery",
            class_confidence=0.87,
            density_g_cm3=1.81,
            colour_histogram=[h * 0.98 for h in hist_pottery],
            uv_signature=0.31,
            inscription_text="[MISSING]ም",
        ),
        ObjectRecord(
            object_id="AXUM-OBJ-004",
            sequence_number=4,
            class_name="coin",
            class_confidence=0.94,
            density_g_cm3=8.71,
            colour_histogram=[0.05] * 16,
        ),
    ]

    for shard in shards:
        grouper.register_object(shard)

    pottery_group = grouper.get_group_for_object("AXUM-OBJ-001")
    coin_group = grouper.get_group_for_object("AXUM-OBJ-004")

    print(f"\nPottery group: {pottery_group.group_id if pottery_group else 'NONE'}")
    if pottery_group:
        print(f"  Confirmed: {pottery_group.confirmed_members}")
        print(f"  Possible:  {pottery_group.possible_members}")
        print(f"  Confidence: {pottery_group.reconstruction_conf:.0%}")
        print(f"  Completeness: {pottery_group.estimated_completeness:.0%}")

    print(f"\nCoin grouped with pottery? {coin_group is not None and coin_group == pottery_group}")

    assert pottery_group is not None, "Three pottery shards should form a group"
    assert len(pottery_group.confirmed_members) >= 2, "At least 2 confirmed members"
    assert coin_group is None or coin_group != pottery_group, "Coin must not join pottery group"

    test_path.unlink(missing_ok=True)
    logger.success("Fragment grouper synthetic test passed")


if __name__ == "__main__":
    _synthetic_test()
