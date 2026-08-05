"""Medical Lang Bridge Tool — Tool + Sampling (doc §2.3, Table 9).

This tool deliberately owns no LLM. It issues ``ctx.session.create_message()``
with ``ModelPreferences`` hints — nova-lite for multilingual content,
command-r-plus for English — and the *calling agent* runs inference through its
own LiteLLM client and returns a ``CreateMessageResult``.

That split is the point of the Sampling primitive: LLM resource management is a
client responsibility, tool logic is a server responsibility. If the tool called
Bedrock itself the primitive would not be demonstrated, so a client without a
sampling callback is a hard error rather than a fallback to local inference.
"""

from __future__ import annotations

from typing import Any

import mcp.types as types
from mcp.server.fastmcp import Context

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import SamplingUnsupportedError
from hospital_ai.core.logging import get_logger
from hospital_ai.mcp_servers.primary.prompts import render

_log = get_logger(__name__, tool="medical-lang-bridge")

#: Script ranges and stopword markers used for language detection. The dataset
#: covers English, Spanish, Hindi, German, French and Dutch.
_SCRIPT_RANGES = {
    "hi": ((0x0900, 0x097F),),  # Devanagari
    "te": ((0x0C00, 0x0C7F),),  # Telugu
}

_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "es": ("paciente", "diagnóstico", "medicamentos", "alta", "seguimiento", "hospital "),
    "de": ("patient", "diagnose", "medikamente", "entlassung", "nachsorge", "krankenhaus"),
    "fr": ("patient", "diagnostic", "médicaments", "sortie", "suivi", "hôpital"),
    "nl": ("patiënt", "diagnose", "medicatie", "ontslag", "controle", "ziekenhuis"),
    "en": ("patient", "diagnosis", "medication", "discharge", "follow-up", "hospital"),
}


def detect_language(text: str) -> tuple[str, float]:
    """Identify the source language and how confident that call is.

    Script detection is decisive for Devanagari. For Latin-script languages the
    marker counts are compared, and a narrow margin lowers the confidence,
    which is what feeds the ``low_translation_confidence`` finding.
    """
    if not text.strip():
        return "en", 0.0

    for language, ranges in _SCRIPT_RANGES.items():
        hits = sum(
            1 for ch in text if any(low <= ord(ch) <= high for low, high in ranges)
        )
        if hits > len(text) * 0.15:
            return language, 0.97

    lowered = text.lower()
    scores = {
        language: sum(lowered.count(marker) for marker in markers)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "en", 0.3

    total = sum(scores.values()) or 1
    share = scores[best] / total
    runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    margin = (scores[best] - runner_up) / scores[best]
    return best, round(min(0.99, 0.55 + 0.45 * (share * 0.5 + margin * 0.5)), 3)


def model_preferences_for(language: str) -> types.ModelPreferences:
    """Advertise the model the client should route to.

    Doc §2.3: "ModelPreferences with model hints (nova-lite for multilingual,
    command-r-plus for English)".
    """
    llm = get_settings().llm
    non_english = llm.sampling_hint_non_english
    english = llm.sampling_hint_english

    if language == "en":
        return types.ModelPreferences(
            hints=[types.ModelHint(name=english)],
            intelligencePriority=0.6,
            speedPriority=0.4,
            costPriority=0.3,
        )
    return types.ModelPreferences(
        hints=[types.ModelHint(name=non_english)],
        intelligencePriority=0.8,
        speedPriority=0.3,
        costPriority=0.2,
    )


def expand_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Expand medical abbreviations using the rules.yaml dictionary."""
    import re

    mapping = (
        get_settings().rules.get("normalization_standards", {}).get("abbreviation_map", {})
    )
    expanded: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        for abbreviation, expansion in mapping.items():
            if token == abbreviation:
                expanded[abbreviation] = expansion
                return expansion
        return token

    if not mapping:
        return text, {}

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(a) for a in sorted(mapping, key=len, reverse=True)) + r")\b"
    )
    return pattern.sub(replace, text), expanded


def score_translation_confidence(
    source: str, translated: str, detection_confidence: float, *, sampled: bool
) -> float:
    """Combine detection certainty with output plausibility.

    A translation that collapses or balloons relative to its source is
    suspicious, and an un-sampled (stubbed) translation can never score above
    the 0.70 threshold — that keeps offline runs honest by routing them to HITL
    rather than silently auto-approving.
    """
    if not translated.strip():
        return 0.0

    ratio = len(translated) / max(len(source), 1)
    if len(source) < 25:
        # Short fields translate to wildly varying lengths ("Man" -> "Male",
        # or a one-line clause the model expands). Length tells us nothing
        # here, so judge them on detection certainty alone.
        length_plausibility = 1.0
    else:
        length_plausibility = 1.0 if 0.5 <= ratio <= 2.5 else max(0.0, 1.0 - abs(1 - ratio) / 3)

    score = 0.5 * detection_confidence + 0.5 * length_plausibility
    if not sampled:
        score = min(score, 0.65)
    return round(max(0.0, min(1.0, score)), 3)


async def translate_via_sampling(
    ctx,
    text: str,
    source_language: str | None = None,
    detection_confidence: float | None = None,
    *,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Translate clinical text by asking the *client* to run inference.

    ``detection_confidence`` lets a caller supply the language certainty it
    measured across the whole record. Re-detecting from a three-character field
    such as a gender value would produce a meaningless score.
    """
    detected, measured = detect_language(text)
    detection_confidence = measured if detection_confidence is None else detection_confidence
    language = source_language or detected

    system_prompt = render("abbreviation-normalization-prompt", source_language=language)
    preferences = model_preferences_for(language)

    sampled = False
    model_used: str | None = None
    translated = text

    if language == "en":
        # Already English: expansion only, no translation round trip.
        translated, _ = expand_abbreviations(text)
        sampled = False
        model_used = "none (source already English)"
    else:
        try:
            result = await ctx.session.create_message(
                messages=[
                    types.SamplingMessage(
                        role="user",
                        content=types.TextContent(type="text", text=text),
                    )
                ],
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                model_preferences=preferences,
            )
            content = result.content
            translated = content.text if isinstance(content, types.TextContent) else str(content)
            model_used = result.model
            sampled = True
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error below
            _log.error(
                "sampling request failed",
                extra={"language": language, "error": str(exc)},
            )
            raise SamplingUnsupportedError(
                "The calling agent did not satisfy the sampling request. The "
                "Medical Lang Bridge Tool has no LLM of its own by design "
                f"(doc §2.3). Underlying error: {exc}"
            ) from exc

    expanded_text, abbreviations = expand_abbreviations(translated)
    confidence = score_translation_confidence(
        text, expanded_text, detection_confidence, sampled=sampled or language == "en"
    )

    threshold = get_settings().rules["quality_thresholds"]["translation_confidence_min"]

    _log.info(
        "language bridge complete",
        extra={
            "detected_language": detected,
            "sampled": sampled,
            "model": model_used,
            "confidence": confidence,
        },
    )
    return {
        "source_language": language,
        "detected_language": detected,
        "detection_confidence": detection_confidence,
        "translated_text": expanded_text,
        "abbreviations_expanded": abbreviations,
        "translation_confidence": confidence,
        "below_threshold": confidence < threshold,
        "threshold": threshold,
        "sampling_used": sampled,
        "model_used": model_used,
        "model_hints": [hint.name for hint in (preferences.hints or [])],
    }


def register(mcp) -> None:
    @mcp.tool(
        name="medical_lang_bridge",
        description=(
            "Translate clinical text to English and expand medical "
            "abbreviations. Issues an MCP Sampling request to the calling "
            "agent's LLM client with nova-lite / command-r-plus model hints; "
            "the tool performs no inference itself."
        ),
    )
    async def medical_lang_bridge(
        ctx: Context,
        text: str,
        source_language: str | None = None,
        detection_confidence: float | None = None,
    ) -> dict[str, Any]:
        return await translate_via_sampling(ctx, text, source_language, detection_confidence)

    @mcp.tool(
        name="detect_clinical_language",
        description="Identify the language of a clinical document and report "
        "the detection confidence.",
    )
    def detect_clinical_language(text: str) -> dict[str, Any]:
        language, confidence = detect_language(text)
        return {"language": language, "confidence": confidence}
