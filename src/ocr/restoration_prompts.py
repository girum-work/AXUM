"""
AXUM ROVER — Shared LLM restoration prompts
=============================================
Single source of truth for system/user prompts used at inference
and for Ge'ez restoration fine-tuning dataset generation.

Why this module exists:
    Training data must match inference prompts exactly. Previously
    prompts lived only in llm_restoration.py while generate_dataset.py
    used a different Alpaca-style template — causing train/serve skew.
"""

from __future__ import annotations

import json

# Few-shot examples injected into every restoration user prompt.
# Based on real Aksumite and Zagwe dynasty inscription patterns.
FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "damaged": "ሰ[MISSING]ም",
        "period": "Aksumite Kingdom, 4th century CE",
        "location": "Aksum, northern Ethiopia",
        "restored": "ሰላም",
        "translation": "Peace / Greeting",
        "reasoning": (
            "ሰላም is the most frequent single-word inscription in Ethiopian "
            "heritage sites. The pattern ሰ_ም with one missing grapheme "
            "almost certainly resolves to ላ."
        ),
    },
    {
        "damaged": "ዓጼ [MISSING]ዛ[MISSING] ነጉሠ ነገሥ[MISSING]",
        "period": "Aksumite Kingdom, 4th century CE",
        "location": "Aksum stelae field",
        "restored": "ዓጼ ዓዛና ነጉሠ ነገሥት",
        "translation": "Emperor Ezana, King of Kings",
        "reasoning": (
            "ዓጼ followed by a name beginning with ዓ and ending in ዛ suggests "
            "ዓዛና (Ezana). ነጉሠ ነገሥት is the standard royal formula."
        ),
    },
    {
        "damaged": "[MISSING]ርያ[MISSING] ወ[MISSING]ድ",
        "period": "Zagwe dynasty, 12th century CE",
        "location": "Lalibela, Beta Maryam church",
        "restored": "ማርያም ወወልድ",
        "translation": "Mary and the Son [of God]",
        "reasoning": (
            "Beta Maryam dedications almost always contain ማርያም. "
            "The pattern _ርያ_ resolves to ማርያም; ወ_ድ is ወልድ."
        ),
    },
    {
        "damaged": "ቅ[MISSING]ስ ጊ[MISSING]ር[MISSING]ስ",
        "period": "Post-Aksumite, 7th-10th century CE",
        "location": "Tigray region church",
        "restored": "ቅዱስ ጊዮርጊስ",
        "translation": "Saint George",
        "reasoning": (
            "ቅ_ስ is almost certainly ቅዱስ. The remaining pattern resolves "
            "to ጊዮርጊስ, the most common named saint in Ethiopian inscriptions."
        ),
    },
]


def build_system_prompt() -> str:
    """
    System prompt for Ge'ez inscription restoration (LM Studio / QLoRA).

    Returns:
        Fixed system string matching production inference.
    """
    return """You are an expert computational epigraphist specialising in \
ancient Ge'ez (Ethiopic) script from Ethiopian heritage sites.

Your task is to restore damaged ancient inscriptions where characters \
are missing (shown as [MISSING]) due to erosion, chipping, or weathering.

Rules you MUST follow:

1. Use knowledge of Ge'ez grammar and historical context.

2. Consider historical period and location.

3. Common inscription words:

ሰላም
ዓጼ
ማርያም
ክርስቶስ
አምላክ
ቅዱስ
ነጉሥ
ጽዮን
ሐሌሉያ
ኢትዮጵያ

4. Aksumite formula:

ዓጼ [name] ነጉሠ ነገሥት

5. Zagwe inscriptions often contain:

ማርያም
ጊዮርጊስ
ክርስቶስ

6. Trinitarian expressions:

አብ
ወልድ
መንፈስ

7. If confidence is low, say so.

Respond ONLY:

{
"restored_text":"",
"translation":"",
"confidence":0.85,
"reasoning":"",
"needs_expert":false
}
"""


def build_user_prompt(
    damaged_text: str,
    period: str = "",
    location: str = "",
    include_few_shot: bool = True,
) -> str:
    """
    User prompt for one restoration request.

    Args:
        damaged_text: OCR or synthetic text with [MISSING] tokens
        period: Historical period context
        location: Site/region context
        include_few_shot: If True, prepend FEW_SHOT_EXAMPLES (inference default)

    Returns:
        User message body string
    """
    examples_text = ""
    if include_few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            examples_text += f"""

Example:

Damaged text:
{ex['damaged']}

Period:
{ex['period']}

Location:
{ex['location']}

Output:

{{
"restored_text":"{ex['restored']}",
"translation":"{ex['translation']}",
"confidence":0.88,
"reasoning":"{ex['reasoning']}",
"needs_expert":false
}}

"""

    prefix = (
        f"\n\nHere are examples of restoration:\n{examples_text}\n\nNow restore:\n"
        if include_few_shot
        else "\n\nRestore:\n"
    )

    return f"""{prefix}
Damaged text:
{damaged_text}

Period:
{period if period else "Unknown"}

Location:
{location if location else "Unknown"}

Respond ONLY with JSON.
"""


def build_assistant_json(
    restored_text: str,
    translation: str,
    confidence: float,
    reasoning: str,
    needs_expert: bool = False,
) -> str:
    """
    Canonical JSON string for the assistant turn in chat fine-tuning.

    Args:
        restored_text: Ground-truth or model target Ge'ez text
        translation: English gloss
        confidence: 0.0–1.0 (deterministic from damage rate in synthetic data)
        reasoning: Short epigraphic justification
        needs_expert: True when damage is too heavy for automation

    Returns:
        JSON string (UTF-8, ensure_ascii=False when dumped by caller)
    """
    return json.dumps(
        {
            "restored_text": restored_text,
            "translation": translation,
            "confidence": round(max(0.0, min(1.0, confidence)), 2),
            "reasoning": reasoning,
            "needs_expert": needs_expert,
        },
        ensure_ascii=False,
    )


def build_chat_messages(
    damaged_text: str,
    restored_text: str,
    translation: str,
    period: str,
    location: str,
    confidence: float,
    reasoning: str,
    needs_expert: bool = False,
    include_few_shot: bool = False,
) -> list[dict]:
    """
    Build OpenAI-style messages list for Unsloth / TRL fine-tuning.

    Training rows use include_few_shot=False to avoid duplicating static
    examples thousands of times; inference uses include_few_shot=True.

    Args:
        damaged_text: Input with [MISSING] tokens
        restored_text: Target restoration
        translation: Target English translation
        period: Historical period metadata
        location: Location metadata
        confidence: Target confidence score
        reasoning: Target reasoning string
        needs_expert: Expert-review flag
        include_few_shot: Whether to embed few-shot block in user message

    Returns:
        List of {role, content} dicts
    """
    return [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt(
                damaged_text, period, location, include_few_shot=include_few_shot
            ),
        },
        {
            "role": "assistant",
            "content": build_assistant_json(
                restored_text, translation, confidence, reasoning, needs_expert
            ),
        },
    ]
