"""
AXUM ROVER — Artefact image label rules
========================================
Shared metadata/filename filters for the classification dataset.

AXUM preserves Ethiopian cultural heritage broadly (not Aksum-only).
These rules reject non-artefact media (paintings, documents) and
re-route mislabelled museum objects (e.g. crosses filed as coins).

Used by:
  - scripts/download_artefact_images.py (pre-save filter)
  - scripts/cleanup_artefact_dataset.py (post-hoc audit of all classes)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# All OBJ_CLASSES for type hints
VALID_CLASSES = (
    "pottery",
    "stone_carving",
    "coin",
    "inscription_fragment",
    "other",
)


class LabelAction(str, Enum):
    """What to do with a candidate image."""

    KEEP = "keep"
    MOVE = "move"
    REJECT = "reject"


@dataclass
class LabelDecision:
    """
    Result of applying label rules to one image's metadata text.

    Attributes:
        action: keep in folder, move to another class, or reject entirely
        target_class: Destination class when action is MOVE
        reason: Human-readable explanation for logs/metadata
    """

    action: LabelAction
    target_class: str | None = None
    reason: str = ""


def _norm(text: str) -> str:
    """Lowercase collapsed text for keyword matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _any_kw(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


# ── Global: never train on these (all classes) ─────────────────
GLOBAL_REJECT_KEYWORDS: tuple[str, ...] = (
    "oil on canvas",
    "oil on panel",
    "oil on wood",
    "tempera on",
    "watercolor",
    "watercolour",
    "fresco",
    "engraving",
    "lithograph",
    "woodcut",
    "etching",
    "illustration",
    "manuscript illumination",
    "book illustration",
    "page from",
    "catalogue plate",
    "catalog plate",
    "diagram",
    "map of",
    "floor plan",
    "poster",
    "postcard",
    "stamp album",
    "screenshot",
    "pdf",
    "adoration of",
    "annunciation",
    "nativity",
    "crucifixion painting",
    "family portrait",
    "portrait of",
    "interior with",
    "landscape painting",
    "still life painting",
    "processional cross",  # often a photo of painting/drawing
    "procession of",
    "triumphal arch",
    "fontana del",  # European fountains, etc.
    "european painting",
    "dutch painting",
    "flemish painting",
    "italian painting",
    "renaissance painting",
    "baroque painting",
    "medieval painting",
    "byzantine icon painting",
    "icon painting",
    "stained glass window",
    "tapestry",
    "mosaic floor",
    "excavation plan",
    "site plan",
    "museum gallery interior",
    "gallery view",
    "crowd at",
    "wedding",
    "ceremony photo",
)

# Non-Ethiopian region cues — reject unless Ethiopia/Horn also mentioned
NON_ETHIOPIA_REGION: tuple[str, ...] = (
    "greek ",
    "roman ",
    "egyptian museum cairo",
    "cairo egyptian museum",
    "british museum greek",
    "minoan",
    "etruscan",
    "etruscan",
    "assyrian",
    "babylonian",
    "mesopotam",
    "cuneiform",
    "gandhara",
    "mayan",
    "aztec",
    "inca ",
    "pre-columbian",
    "chinese ",
    "japanese ",
    "korean ",
    "celtic ",
    "viking",
    "scythian",
    "etruscan",
    "persian miniature",
    "indian sculpture",
    "louvre",
    "versailles",
    "vatican",
    "uffizi",
    "rijksmuseum",
    "hermitage",
)

ETHIOPIA_POSITIVE: tuple[str, ...] = (
    "ethiopia",
    "ethiopian",
    "abyssin",
    "axum",
    "aksum",
    "aksumite",
    "lalibela",
    "tigray",
    "tigrai",
    "amhara",
    "gondar",
    "gonder",
    "harar",
    "oromo",
    "geez",
    "ge'ez",
    "ethiopic",
    "ezana",
    "kaleb",
    "zagwe",
    "solomonic",
    "horn of africa",
    "east africa",
    "nubia",
    "nubian ethiop",
    "menelik",
    "menelek",
    "menilek",
    "haile selassie",
    "haile selassie",
    "selassie",
    "national bank of ethiopia",
    "national museum of ethiopia",
    "institute of ethiopian",
    "birr",
    "sante",
    "cent",
    "ethiopian currency",
    "ethiopian coin",
    "addis ababa",
    "shewa",
    "wollo",
    "derg",
)

# ── Class-specific signals ─────────────────────────────────────
COIN_POSITIVE = (
    "coin",
    "coins",
    "numismat",
    "obverse",
    "reverse",
    "denarius",
    "drachm",
    "solidus",
    "currency",
    "medallion coin",
    "coinage",
    "birr",
    "cent",
    "sante",
    "menelik",
    "menelek",
    "haile selassie",
    "aksumite coin",
    "axum coin",
    "ethiopian coin",
    "ethiopia coin",
)

# Foreign currency — reject even if search noise lands in coin folder
FOREIGN_COIN_REJECT = (
    "us quarter",
    "american coin",
    "euro coin",
    "british penny",
    "roman coin",
    "byzantine coin",
    "greek drachm",
    "celtic coin",
)

COIN_REJECT = (
    "pendant cross",
    "pendant_cr",
    "pendant cr",
    "hand cross",
    "processional cross",
    "neck cross",
    "pectoral cross",
    "cross pendant",
    "pendant cross",
    "brooklyn museum 79.72",  # Ethiopian crosses accession series
    "79_72_",
    "79.72.",
    "crucifix pendant",
    "rosary",
    "icon cross",
    "wall cross",
)

CROSS_OTHER_POSITIVE = (
    "cross",
    "crucifix",
    "pendant",
    "processional",
    "hand cross",
    "neck cross",
    "ethiopian cross",
)

INSCRIPTION_POSITIVE = (
    "inscription",
    "inscribed",
    "epigraph",
    "stele inscription",
    "stela inscription",
    "geez",
    "ge'ez",
    "ethiopic script",
    "ostracon",
    "ostraka",
    "tablet inscription",
    "stone tablet",
)

INSCRIPTION_REJECT = (
    "coin with",
    "gold coin",
    "silver coin",
    "coinage",
    "fontana",
    "fountain",
    "triumph",
    "procession",
    "manuscript page",
    "book page",
    "folio",
)

POTTERY_POSITIVE = (
    "pottery",
    "ceramic",
    "vessel",
    "amphora",
    "bowl",
    "jar",
    "urn",
    "terracotta",
    "potsherd",
    "sherd",
)

STONE_POSITIVE = (
    "stele",
    "stela",
    "obelisk",
    "relief",
    "carving",
    "sculpture stone",
    "rock-hewn",
    "stone carving",
    "relief carving",
)


def is_ethiopia_related(text: str) -> bool:
    """
    True if metadata suggests Ethiopian / Horn of Africa heritage.

    Args:
        text: Combined title, filename, notes, Met culture fields

    Returns:
        Whether Ethiopian context is present
    """
    t = _norm(text)
    return _any_kw(t, ETHIOPIA_POSITIVE)


# Legacy Met downloads used broad non-Ethiopian search strings
LEGACY_MET_BROAD_QUERIES: tuple[str, ...] = (
    "pottery vessel ancient",
    "amphora",
    "ceramic vase",
    "greek vase",
    "roman pottery",
    "stone sculpture ancient",
    "stela",
    "relief stone",
    "coin ancient",
    "gold coin",
    "roman coin",
    "byzantine coin",
    "epigraphy",
    "tablet inscription",
    "antiquity bronze",
    "archaeological",
    "ancient jewelry",
)


def is_legacy_met_pollution(text: str) -> tuple[bool, str]:
    """
    Reject images from old Met API runs that ignored Ethiopian scope.

    Args:
        text: metadata notes + filename

    Returns:
        (is_pollution, reason)
    """
    t = _norm(text)
    if "met query=" not in t and not t.startswith("met_"):
        return False, ""
    if is_ethiopia_related(t):
        return False, ""
    for q in LEGACY_MET_BROAD_QUERIES:
        if f"met query='{q}'" in t or f'met query="{q}"' in t:
            return True, f"legacy_met_broad_query:{q}"
    if t.startswith("met_") and "ethiop" not in t and "aksum" not in t:
        return True, "legacy_met_no_ethiopia_in_meta"
    return False, ""


def is_globally_rejected(text: str) -> tuple[bool, str]:
    """
    Reject paintings, prints, diagrams, and non-artefact photos globally.

    Args:
        text: Combined metadata string

    Returns:
        (should_reject, reason)
    """
    t = _norm(text)
    for kw in GLOBAL_REJECT_KEYWORDS:
        if kw in t:
            return True, f"global_reject:{kw}"

    # Non-Ethiopia region without Ethiopian offset
    if _any_kw(t, NON_ETHIOPIA_REGION) and not is_ethiopia_related(t):
        for kw in NON_ETHIOPIA_REGION:
            if kw in t:
                return True, f"non_ethiopia_region:{kw.strip()}"

    return False, ""


def decide_label(
    text: str,
    current_class: str,
    *,
    require_ethiopia: bool = True,
) -> LabelDecision:
    """
    Decide keep, move, or reject for one image given folder + metadata.

    Args:
        text: Filename + title + notes + API fields concatenated
        current_class: Folder the image is in (or download target)
        require_ethiopia: If True, reject items with no Ethiopian context
            unless they clearly match class-specific positives (coins etc.)

    Returns:
        LabelDecision with action and optional target_class
    """
    t = _norm(text)
    if not t:
        return LabelDecision(LabelAction.KEEP, reason="no_text")

    rejected, reason = is_globally_rejected(text)
    if rejected:
        return LabelDecision(LabelAction.REJECT, reason=reason)

    legacy, leg_reason = is_legacy_met_pollution(text)
    if legacy:
        return LabelDecision(LabelAction.REJECT, reason=leg_reason)

    eth = is_ethiopia_related(text)

    # ── Cross / pendant misfiled as coin (common Wikimedia noise) ──
    if _any_kw(t, COIN_REJECT) or (
        "cross" in t and "coin" not in t and current_class == "coin"
    ):
        if _any_kw(t, CROSS_OTHER_POSITIVE) or "pendant" in t:
            return LabelDecision(
                LabelAction.MOVE,
                target_class="other",
                reason="cross_or_pendant_not_coin",
            )

    # Coin class: Ethiopian currency any era (ancient → modern), not crosses
    if current_class == "coin":
        if _any_kw(t, FOREIGN_COIN_REJECT) and not eth:
            return LabelDecision(LabelAction.REJECT, reason="foreign_coin")
        if not _any_kw(t, COIN_POSITIVE):
            if _any_kw(t, CROSS_OTHER_POSITIVE):
                return LabelDecision(
                    LabelAction.MOVE,
                    target_class="other",
                    reason="cross_not_coin",
                )
            if require_ethiopia and not eth:
                return LabelDecision(
                    LabelAction.REJECT,
                    reason="coin_not_ethiopia_or_numismatic",
                )
            return LabelDecision(LabelAction.REJECT, reason="not_coin_like")

    # Inscription: reject coins and European scenes
    if current_class == "inscription_fragment":
        if _any_kw(t, INSCRIPTION_REJECT):
            return LabelDecision(LabelAction.REJECT, reason="inscription_noise")
        if _any_kw(t, ("coin", "coinage")) and "inscription" not in t:
            return LabelDecision(
                LabelAction.MOVE,
                target_class="coin",
                reason="coin_misfiled_as_inscription",
            )
        if require_ethiopia and not eth and not _any_kw(t, INSCRIPTION_POSITIVE):
            return LabelDecision(
                LabelAction.REJECT,
                reason="inscription_not_ethiopia",
            )

    # Pottery / stone: require Ethiopian context (blocks generic Met "ancient" pulls)
    if current_class in ("pottery", "stone_carving", "coin", "inscription_fragment"):
        if require_ethiopia and not eth:
            class_pos = {
                "pottery": POTTERY_POSITIVE,
                "stone_carving": STONE_POSITIVE,
                "coin": COIN_POSITIVE,
                "inscription_fragment": INSCRIPTION_POSITIVE,
            }.get(current_class, ())
            if not _any_kw(t, class_pos):
                return LabelDecision(
                    LabelAction.REJECT,
                    reason=f"{current_class}_non_ethiopia",
                )

    # Other: crosses and metalwork OK; still reject global junk
    if current_class == "other":
        if require_ethiopia and not eth and not _any_kw(
            t, CROSS_OTHER_POSITIVE + ("bronze", "metal", "jewelry", "artefact")
        ):
            return LabelDecision(
                LabelAction.REJECT,
                reason="other_non_ethiopia",
            )

    return LabelDecision(LabelAction.KEEP, reason="ok")


def build_met_context(obj: dict) -> str:
    """
    Extract searchable text from a Met Open Access object JSON.

    Args:
        obj: Met API object dict

    Returns:
        Combined string for decide_label()
    """
    parts = [
        obj.get("title", ""),
        obj.get("objectName", ""),
        obj.get("culture", ""),
        obj.get("period", ""),
        obj.get("dynasty", ""),
        obj.get("reign", ""),
        obj.get("classification", ""),
        obj.get("department", ""),
        str(obj.get("objectID", "")),
    ]
    for tag in obj.get("tags") or []:
        if isinstance(tag, dict):
            parts.append(tag.get("term", ""))
    return " ".join(p for p in parts if p)


def filename_to_title_hint(filename: str) -> str:
    """
    Turn wiki_File_Brooklyn_Museum_79_72_11_Pendant_Cr_*.jpg into readable tokens.

    Args:
        filename: On-disk image name

    Returns:
        Spaced lowercase hint string
    """
    base = filename.rsplit(".", 1)[0]
    base = re.sub(r"^(wiki_|met_|si_)", "", base)
    base = base.replace("File_", "").replace("_", " ")
    return base.lower()
