"""
AXUM ROVER — Ge'ez LLM Contextual Restoration Engine
======================================================
Bridges the gap between raw OCR output and historically accurate text.

The problem this solves:
    Ancient Ge'ez inscriptions on stone artefacts are frequently:
    - Eroded (characters worn away by weather)
    - Chipped (physical damage to the stone surface)
    - Partially obscured (mineral deposits, dirt)

    A standard OCR model outputs gibberish for these missing characters.
    This engine replaces that gibberish with linguistically and
    historically informed predictions using a local LLM.

Pipeline:
    CRNN OCR output:  "ሰ[MISSING]ም ለ[MISSING][MISSING]"
    LLM restoration:  "ሰላም ለኢትዮጵያ"  (Peace to Ethiopia)
    Translation:      "Peace to Ethiopia"

The LLM runs entirely locally via LM Studio — no internet, no API key,
no data leaves the machine. Critical for museum-environment privacy.

Two operational modes:
    ollama_fewshot:  Uses Ollama with a few-shot prompt containing
                     real historical Ge'ez examples. Best accuracy.
                     Requires: ollama installed + model pulled.

    rule_based:      Pure Python fallback using a Ge'ez linguistic
                     pattern database. No LLM needed. Lower accuracy
                     but always available, even offline without Ollama.

Author: Axum Rover Team
"""

import re
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, LM_STUDIO_BASE_URL, LM_STUDIO_TIMEOUT_SEC, COMPUTE_TIER, RESTORATION_MODEL_GPU
from src.ocr.restoration_prompts import (
    FEW_SHOT_EXAMPLES,
    build_system_prompt,
    build_user_prompt,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class RestorationResult:
    """
    Complete result from one restoration attempt.
    Passed to the dashboard and catalogue generator.
    """
    # Input
    raw_ocr_text:     str   = ""   # original OCR output with [MISSING] tokens
    artifact_period:  str   = ""   # e.g. "Aksumite, 4th century CE"
    artifact_location:str   = ""   # e.g. "Lalibela, Beta Maryam church"

    # Output
    restored_text:    str   = ""   # predicted complete Ge'ez text
    translation:      str   = ""   # English translation
    confidence:       float = 0.0  # 0.0 to 1.0
    reasoning:        str   = ""   # why the LLM made this prediction
    missing_count:    int   = 0    # number of [MISSING] tokens filled
    mode_used:        str   = ""   # 'ollama_fewshot' or 'rule_based'

    # Flags
    is_known_phrase:  bool  = False  # matches a known inscription pattern
    matched_pattern:  str   = ""     # which pattern it matched
    needs_expert:     bool  = False  # too uncertain for automated output


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — GE'EZ LINGUISTIC KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════

# Common Ge'ez inscription phrases with their translations.
# Sourced from: Aksumite royal inscriptions, Lalibela dedications,
# Ethiopian Orthodox liturgical texts, Kebra Nagast excerpts.
# Each entry: (pattern_regex, completed_form, translation, period, context)
KNOWN_PHRASES = [
    # Greeting / dedicatory formulas
    (r"ሰ.ም",         "ሰላም",          "Peace / Greeting",
     "All periods",   "Most common word in Ethiopian inscriptions"),

    (r"ሰ.ም ለ.+",     "ሰላም ለ...",     "Peace to [recipient]",
     "Aksumite+",     "Dedicatory formula"),

    (r"ዓጼ .+",        "ዓጼ ...",       "Emperor [name]",
     "Aksumite",      "Royal title — Aksumite kings"),

    (r"ን.ሥ",          "ነጉሥ",          "King",
     "All periods",   "Royal designation"),

    (r"ማ.ያም",        "ማርያም",        "Mary (Virgin Mary)",
     "Post-4th CE",   "Christian dedication, common at Lalibela"),

    (r"ክ.ስቶ.",        "ክርስቶስ",       "Christ",
     "Post-4th CE",   "Christian inscription"),

    (r"አ.ላክ",         "አምላክ",        "God",
     "All periods",   "Theological inscription"),

    (r"ቅ.ስ",          "ቅዱስ",         "Holy / Saint",
     "Post-4th CE",   "Hagiographic inscription"),

    (r"ኢ.ዮ.ያ",        "ኢትዮጵያ",       "Ethiopia",
     "All periods",   "National/geographic reference"),

    (r"ጽ.ዮን",         "ጽዮን",         "Zion / Sacred Ark",
     "Aksumite+",     "Religious/royal — Ark of the Covenant"),

    (r"ሐ.ሌ",          "ሐሌሉያ",        "Hallelujah / Praise God",
     "Post-4th CE",   "Liturgical inscription"),

    (r"ወ.ድ",          "ወልድ",         "Son (of God)",
     "Post-4th CE",   "Trinitarian formula"),

    (r"አ.ብ",          "አብ",          "Father (God the Father)",
     "Post-4th CE",   "Trinitarian formula"),

    (r"መ.ፍ.ስ",        "መንፈስ",        "Spirit (Holy Spirit)",
     "Post-4th CE",   "Trinitarian formula"),

    (r"ዘ.ወ.ር",        "ዘውር",         "Crown / Royalty",
     "Aksumite",      "Royal inscription"),

    (r"ሃ.ማ.ት",        "ሃይማኖት",       "Faith / Religion",
     "Post-4th CE",   "Religious dedication"),

    (r"ደ.ቅ",          "ደቅ",          "Children / Disciples",
     "Post-4th CE",   "Monastic/religious context"),

    (r"ብ.ሔ.ቤ",        "ብሔረ ቤተ",      "Land of the House [of God]",
     "Aksumite+",     "Geographical/religious"),

    (r"ዮ.ሐ.ስ",        "ዮሐንስ",        "John (Apostle / Saint)",
     "Post-4th CE",   "Hagiographic — Saint John"),

    (r"ጊ.ዮ.ጊ.",        "ጊዮርጊስ",      "George (Saint George)",
     "Post-5th CE",   "Patron saint inscription — very common"),
]


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — FEW-SHOT EXAMPLES FOR THE LLM PROMPT
# ═══════════════════════════════════════════════════════════════

# FEW_SHOT_EXAMPLES imported from restoration_prompts (shared with dataset generator)


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — RULE-BASED FALLBACK ENGINE
# ═══════════════════════════════════════════════════════════════

class RuleBasedRestorer:
    """
    Pure Python restoration using pattern matching.
    No LLM required. Used when Ollama is unavailable.

    Accuracy is lower than the LLM engine but always works.
    Sufficient for demo when the LLM is warming up or offline.
    """

    def __init__(self):
        self.phrases = KNOWN_PHRASES
        logger.info("RuleBasedRestorer initialized "
                    f"({len(self.phrases)} patterns)")

    def restore(self, damaged_text: str) -> dict:
        """
        Attempt to restore damaged text using pattern matching.

        Args:
            damaged_text: OCR output with [MISSING] tokens

        Returns:
            dict with restored_text, translation, confidence, matched_pattern
        """
        # Convert [MISSING] to a single wildcard character for regex
        # [MISSING] can represent one OR more unknown characters
        searchable = damaged_text.replace("[MISSING]", ".")

        best_match      = None
        best_confidence = 0.0

        for pattern, completed, translation, period, context in self.phrases:
            try:
                if re.search(pattern, searchable):
                    # Score by how many characters were actually matched
                    # vs how many were wildcards
                    total_chars   = len(searchable.replace(".", ""))
                    wildcard_count = searchable.count(".")
                    if total_chars > 0:
                        confidence = 1.0 - (wildcard_count /
                                            (total_chars + wildcard_count))
                        confidence = max(0.35, confidence)
                    else:
                        confidence = 0.35

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (completed, translation,
                                      period, context, pattern)
            except re.error:
                continue

        if best_match:
            completed, translation, period, context, pattern = best_match
            return {
                "restored_text":    completed,
                "translation":      translation,
                "confidence":       best_confidence,
                "reasoning":        f"Pattern match: {context} ({period})",
                "matched_pattern":  pattern,
                "is_known_phrase":  True
            }

        # No match found
        # Return original with [MISSING] replaced by ◌ (placeholder)
        cleaned = damaged_text.replace("[MISSING]", "◌")
        return {
            "restored_text":    cleaned,
            "translation":      "[Unrecognised inscription — expert review needed]",
            "confidence":       0.1,
            "reasoning":        "No pattern match found in linguistic database",
            "matched_pattern":  "",
            "is_known_phrase":  False
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — LM STUDIO LLM ENGINE
# ═══════════════════════════════════════════════════════════════

import httpx
from openai import OpenAI


def _make_lm_studio_http_client() -> httpx.Client:
    """
    Build an httpx client that ignores system HTTP_PROXY settings.

    On Windows, Cursor/Clash/VPN proxies often route localhost through
    127.0.0.1:<proxy_port>, which breaks LM Studio at :1234. trust_env=False
    forces a direct connection to the local server.
    """
    return httpx.Client(trust_env=False, timeout=LM_STUDIO_TIMEOUT_SEC)


class LMStudioRestorationEngine:
    """
    LLM-powered restoration using LM Studio.

    Setup:

    1. Install LM Studio
       https://lmstudio.ai/

    2. Download/load model:
         qwen2.5-1.5b-instruct
         qwen2.5-3b-instruct
         llama3.2-3b-instruct
         phi-3-mini

    3. Start local server:

       LM Studio
       -> Developer
       -> Start Server

       Default:
       http://localhost:1234

    4. Install dependency:

       pip install openai
    """

    PREFERRED_MODELS = [

        "qwen2.5-1.5b",

        "qwen2.5-3b",

        "llama3.2-1b",

        "llama3.2-3b",

        "phi-3"
    ]


    def __init__(
        self,
        model_name: str = None
    ):

        self.model_name = model_name
        self.available = False
        self.client = None

        self._initialize()


    def _try_gpu_tier_model(self, available_names: list) -> bool:
        """
        WHAT: if COMPUTE_TIER=gpu and RESTORATION_MODEL_GPU names a
        specific model that's actually loaded in LM Studio, prefer it
        over the generic PREFERRED_MODELS auto-detect list.

        WHY this exists as a separate check rather than just adding
        RESTORATION_MODEL_GPU to PREFERRED_MODELS: PREFERRED_MODELS is a
        substring-matched list used regardless of tier -- mixing a
        GPU-only preference into it would mean a CPU-tier machine that
        happens to have a similarly-named model loaded could pick it up
        by accident. This keeps the GPU-tier preference conditional on
        COMPUTE_TIER actually being "gpu".

        *** SCOPE NOTE, important: this only selects WHICH model name to
        request. Actual GPU acceleration (how many layers get offloaded)
        is an LM Studio server-side runtime setting, configured manually
        in LM Studio's own UI when the model is loaded there -- it is NOT
        something this HTTP client can control per-request via the chat
        completions API. Whoever loads the GPU-tier model in LM Studio
        needs to enable GPU offload there manually; this code only makes
        sure the RIGHT model name gets requested. ***

        Returns True (and sets self.model_name) if a GPU-tier model was
        found and selected, False otherwise (caller falls through to the
        existing PREFERRED_MODELS auto-detect, unchanged).
        """
        if COMPUTE_TIER != "gpu" or not RESTORATION_MODEL_GPU:
            return False
        if RESTORATION_MODEL_GPU in available_names:
            self.model_name = RESTORATION_MODEL_GPU
            logger.info(f"GPU-tier model selected: {self.model_name}")
            return True
        logger.warning(
            f"COMPUTE_TIER=gpu and RESTORATION_MODEL_GPU={RESTORATION_MODEL_GPU!r} "
            f"set, but that model isn't currently loaded in LM Studio "
            f"(available: {available_names}) — falling through to normal "
            f"auto-detect. Load it in LM Studio first if GPU-tier is intended."
        )
        return False

    def _initialize(self):

        """
        Connect to LM Studio and
        auto-detect loaded models.
        """

        try:

            http_client = _make_lm_studio_http_client()
            self.client = OpenAI(
                api_key="lm-studio",
                base_url=LM_STUDIO_BASE_URL,
                http_client=http_client,
            )

            models = self.client.models.list()

            available_names = [
                m.id for m in models.data
            ]

            logger.info(
                f"LM Studio models: "
                f"{available_names}"
            )

            if (
                self.model_name
                and
                self.model_name in available_names
            ):

                self.available = True

                logger.info(
                    f"Using specified model:"
                    f"{self.model_name}"
                )

            elif self._try_gpu_tier_model(available_names):
                self.available = True

            else:

                for preferred in self.PREFERRED_MODELS:

                    for model in available_names:

                        if preferred.lower() in model.lower():

                            self.model_name = model

                            self.available = True

                            logger.info(
                                f"Auto-selected: "
                                f"{model}"
                            )

                            break

                    if self.available:
                        break

            if not self.available:

                logger.warning(
                    "No suitable LM Studio "
                    "model loaded."
                )

        except ImportError:

            logger.warning(
                "openai package missing.\n"
                "Run: pip install openai"
            )

        except Exception as e:

            hint = (
                "Did you start the server in LM Studio (Developer → Server)?"
            )
            if "proxy" in str(e).lower():
                hint += (
                    " Windows proxy may be intercepting localhost — "
                    "this build bypasses that; restart Python and retry."
                )
            logger.warning(f"LM Studio unavailable: {e}\n{hint}")


    def _build_system_prompt(self) -> str:
        """System prompt (shared with fine-tuning dataset generator)."""
        return build_system_prompt()

    def _build_few_shot_prompt(
        self,
        damaged_text: str,
        period: str,
        location: str,
    ) -> str:
        """User prompt with few-shot examples (inference default)."""
        return build_user_prompt(
            damaged_text, period, location, include_few_shot=True
        )


    def restore(
        self,
        damaged_text:str,
        period:str="",
        location:str=""
    ) -> dict:

        """
        Main restoration pipeline.
        """

        if not self.available:

            return self._unavailable_result(
                damaged_text
            )

        system_prompt = (
            self._build_system_prompt()
        )

        user_prompt = (
            self._build_few_shot_prompt(
                damaged_text,
                period,
                location
            )
        )

        logger.info(
            f"LM restoration: "
            f"{damaged_text[:40]}"
        )

        start_time = time.time()

        try:

            response = (
                self.client.chat.completions.create(

                    model=self.model_name,

                    temperature=0.2,

                    top_p=0.9,

                    max_tokens=300,

                    messages=[

                        {
                            "role":"system",
                            "content":
                            system_prompt
                        },

                        {
                            "role":"user",
                            "content":
                            user_prompt
                        }

                    ]
                )
            )

            duration = (
                time.time()
                -
                start_time
            )

            raw_response = (

                response
                .choices[0]
                .message.content
                .strip()
            )

            logger.debug(
                f"LLM response "
                f"({duration:.1f}s): "
                f"{raw_response[:100]}"
            )

            return self._parse_response(
                raw_response,
                damaged_text
            )

        except Exception as e:

            logger.error(
                f"Restoration failed: {e}"
            )

            return self._error_result(
                damaged_text,
                str(e)
            )


    def _parse_response(
        self,
        raw:str,
        original:str
    ) -> dict:

        """
        Parse LM response safely.
        """

        raw = re.sub(
            r"```json\s*",
            "",
            raw
        )

        raw = re.sub(
            r"```\s*",
            "",
            raw
        )

        raw = raw.strip()

        json_match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL
        )

        if not json_match:

            logger.warning(
                f"No JSON found:"
                f"{raw[:100]}"
            )

            return self._error_result(
                original,
                "No JSON found"
            )

        try:

            data = json.loads(
                json_match.group()
            )

            restored = data.get(
                "restored_text",
                ""
            )

            translation = data.get(
                "translation",
                ""
            )

            confidence = float(
                data.get(
                    "confidence",
                    .5
                )
            )

            reasoning = data.get(
                "reasoning",
                ""
            )

            needs_expert = bool(
                data.get(
                    "needs_expert",
                    False
                )
            )

            original_chars = (
                original.replace(
                    "[MISSING]",
                    ""
                )
            )

            if (
                len(restored)
                <
                len(original_chars)*0.5
            ):

                logger.warning(
                    "Suspicious output"
                )

                confidence=min(
                    confidence,
                    0.4
                )

                needs_expert=True


            return {

                "restored_text":
                restored,

                "translation":
                translation,

                "confidence":
                max(
                    0,
                    min(
                        1,
                        confidence
                    )
                ),

                "reasoning":
                reasoning,

                "needs_expert":
                needs_expert,

                "is_known_phrase":
                False,

                "matched_pattern":
                ""
            }

        except Exception as e:

            return self._error_result(
                original,
                str(e)
            )


    def _unavailable_result(
        self,
        original:str
    ) -> dict:

        cleaned=original.replace(
            "[MISSING]",
            "◌"
        )

        return {

            "restored_text":
            cleaned,

            "translation":
            "[LM Studio unavailable]",

            "confidence":
            0.0,

            "reasoning":
            "LM Studio unavailable",

            "needs_expert":
            True,

            "is_known_phrase":
            False,

            "matched_pattern":
            ""
        }


    def _error_result(
        self,
        original:str,
        error:str
    ) -> dict:

        cleaned=original.replace(
            "[MISSING]",
            "◌"
        )

        return {

            "restored_text":
            cleaned,

            "translation":
            "[Restoration failed]",

            "confidence":
            0.0,

            "reasoning":
            f"Error: {error}",

            "needs_expert":
            True,

            "is_known_phrase":
            False,

            "matched_pattern":
            ""
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RESTORATION ENGINE (combines both)
# ═══════════════════════════════════════════════════════════════

class GeezRestorationEngine:
    """
    Main entry point for the Ge'ez restoration pipeline.

    Combines LLM and rule-based engines with automatic fallback:

    1. If OCR has no [MISSING] tokens → return as-is (no restoration needed)
    2. If LLM available → use OllamaRestorationEngine (primary)
    3. If LLM unavailable → use RuleBasedRestorer (fallback)
    4. Compare both results if both available → return higher confidence

    Usage:
        engine = GeezRestorationEngine()
        result = engine.restore(
            damaged_text   = "ሰ[MISSING]ም ለ[MISSING]ትዮጵያ",
            period         = "Aksumite, 4th century CE",
            location       = "Aksum, northern Ethiopia",
            artifact_class = "stone_carving"
        )
        print(result.restored_text)
        print(result.translation)
        print(f"Confidence: {result.confidence:.0%}")
    """

    def __init__(
        self,
        mode:       str = "auto",
        model_name: str = None
    ):
        """
        Args:
            mode: 'auto'         = try LLM first, fall back to rule-based
                  'ollama_fewshot' = LLM only (fail if unavailable)
                  'rule_based'   = rule-based only (no LLM)
            model_name: Specific Ollama model (None = auto-select)
        """
        self.mode = mode

        # Initialize engines
        self.rule_engine = RuleBasedRestorer()

        if mode in ("auto", "ollama_fewshot", "lm_studio"):
            self.llm_engine = LMStudioRestorationEngine(model_name)
        else:
            self.llm_engine = None

        logger.info(
            f"GeezRestorationEngine ready | "
            f"mode={mode} | "
            f"LLM={'available' if (self.llm_engine and self.llm_engine.available) else 'unavailable'}"
        )

    def restore(
        self,
        damaged_text:    str,
        period:          str = "",
        location:        str = "",
        artifact_class:  str = ""
    ) -> RestorationResult:
        """
        Restore a damaged Ge'ez inscription.

        Args:
            damaged_text:   OCR output with [MISSING] tokens
            period:         Historical period (e.g. "Aksumite, 4th CE")
            location:       Where the artefact was found
            artifact_class: Object type from classifier (e.g. "stone_carving")

        Returns:
            RestorationResult dataclass
        """
        missing_count = damaged_text.count("[MISSING]")

        # No restoration needed if no missing tokens
        if missing_count == 0:
            return RestorationResult(
                raw_ocr_text      = damaged_text,
                artifact_period   = period,
                artifact_location = location,
                restored_text     = damaged_text,
                translation       = self._quick_translate(damaged_text),
                confidence        = 0.95,
                reasoning         = "No missing characters — direct OCR output",
                missing_count     = 0,
                mode_used         = "none_needed",
                is_known_phrase   = False,
                needs_expert      = False
            )

        logger.info(
            f"Restoring: '{damaged_text[:50]}' "
            f"({missing_count} missing tokens)"
        )

        # Build context string for LLM
        context_period = period or "Unknown period"
        if artifact_class:
            context_period = f"{context_period} ({artifact_class})"

        # ── Try LLM first ─────────────────────────────────────
        llm_result   = None
        rule_result  = None

        if self.llm_engine and self.llm_engine.available:
            llm_result = self.llm_engine.restore(
                damaged_text, context_period, location
            )
            logger.info(
                f"LLM result: '{llm_result['restored_text'][:40]}' "
                f"conf={llm_result['confidence']:.0%}"
            )

        # ── Always run rule-based for comparison ──────────────
        rule_result = self.rule_engine.restore(damaged_text)

        # ── Select best result ─────────────────────────────────
        if llm_result and rule_result:
            # Use LLM if significantly more confident
            # Use rule-based if it found a known phrase (very reliable)
            if rule_result.get("is_known_phrase") and \
               rule_result["confidence"] >= llm_result["confidence"] - 0.1:
                chosen = rule_result
                mode   = "rule_based"
                logger.info("Selected rule-based result (known phrase match)")
            elif llm_result["confidence"] >= rule_result["confidence"]:
                chosen = llm_result
                mode   = "lm_studio_fewshot"
            else:
                chosen = rule_result
                mode   = "rule_based"
        elif llm_result:
            chosen = llm_result
            mode   = "lm_studio_fewshot"
        elif rule_result:
            chosen = rule_result
            mode   = "rule_based"
        else:
            # This should never happen — rule engine always returns something
            chosen = {
                "restored_text":   damaged_text.replace("[MISSING]", "◌"),
                "translation":     "[Restoration failed]",
                "confidence":      0.0,
                "reasoning":       "All engines failed",
                "needs_expert":    True,
                "is_known_phrase": False,
                "matched_pattern": ""
            }
            mode = "failed"

        return RestorationResult(
            raw_ocr_text      = damaged_text,
            artifact_period   = period,
            artifact_location = location,
            restored_text     = chosen["restored_text"],
            translation       = chosen["translation"],
            confidence        = chosen["confidence"],
            reasoning         = chosen["reasoning"],
            missing_count     = missing_count,
            mode_used         = mode,
            is_known_phrase   = chosen.get("is_known_phrase", False),
            matched_pattern   = chosen.get("matched_pattern", ""),
            needs_expert      = chosen.get("needs_expert", False)
        )

    def _quick_translate(self, text: str) -> str:
        """Quick word-by-word translation for already-complete text."""
        # Use the rule engine's phrase list for word lookup
        translations = []
        for word in text.split():
            found = False
            for pattern, completed, translation, _, _ in KNOWN_PHRASES:
                if re.match(pattern + "$", word):
                    translations.append(translation)
                    found = True
                    break
            if not found:
                translations.append(word)   # keep original if unknown
        return " / ".join(translations) if translations else text

    def batch_restore(
        self,
        damaged_texts: list,
        period:        str = "",
        location:      str = ""
    ) -> list:
        """
        Restore multiple damaged text segments at once.
        Useful when an artefact has several inscription regions.

        Args:
            damaged_texts: List of OCR strings with [MISSING] tokens
            period:        Shared historical period for all segments
            location:      Shared location for all segments

        Returns:
            List of RestorationResult objects
        """
        results = []
        for i, text in enumerate(damaged_texts):
            logger.info(f"Batch restore {i+1}/{len(damaged_texts)}")
            result = self.restore(text, period, location)
            results.append(result)
            # Small delay between LLM calls to prevent overheating on CPU
            if self.llm_engine and self.llm_engine.available:
                time.sleep(0.5)
        return results

    def is_llm_available(self) -> bool:
        """Check if LLM engine is ready."""
        return bool(self.llm_engine and self.llm_engine.available)

    def get_status(self) -> dict:
        """Return status dict for dashboard display."""
        return {
            "mode":          self.mode,
            "llm_available": self.is_llm_available(),
            "llm_model":     self.llm_engine.model_name
                             if self.llm_engine else None,
            "rule_patterns": len(self.rule_engine.phrases)
        }


# Backward-compatible alias (docs previously referenced Ollama)
OllamaRestorationEngine = LMStudioRestorationEngine