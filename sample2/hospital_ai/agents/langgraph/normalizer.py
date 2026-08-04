"""Clinical Normalizer Agent — LangGraph, port 8102 (doc §2.3).

Detects the source language, translates clinical content into English through
**MCP Sampling**, expands medical abbreviations, and reports a translation
confidence score.

This agent is the client half of the Sampling primitive. The Medical Lang
Bridge Tool on the MCP server issues ``ctx.session.create_message()`` with
``ModelPreferences``; :func:`build_sampling_handler` here reads those hints,
routes to the matching LiteLLM model, and returns a ``CreateMessageResult``.
The tool never touches an LLM, and this agent never decides *what* to
translate — that separation is the primitive's architectural intent.

Graph shape::

    detect -> (route) -> translate -> expand -> score -> emit

The route edge skips translation for English records, which still pass through
abbreviation expansion.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import mcp.types as types
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.llm.gateway import LLMGateway, get_gateway

_log = get_logger(__name__, agent="clinical-normalizer")


def _merge(left: list, right: list) -> list:
    return [*left, *right]


class NormalizerState(TypedDict, total=False):
    case_id: str
    patient_id: str
    trace_id: str
    record: dict[str, Any]
    source_language: str
    detected_language: str
    detection_confidence: float
    translated: dict[str, str]
    per_document_confidence: dict[str, float]
    confidence_samples: list[tuple[float, int]]
    abbreviations: dict[str, str]
    translation_confidence: float
    model_used: str | None
    sampling_used: bool
    provenance: str
    warnings: Annotated[list[str], _merge]


def build_sampling_handler(gateway: LLMGateway | None = None):
    """The client-side sampling callback the specification requires.

    Reads the server's ``ModelPreferences`` hint, performs inference through
    LiteLLM, and returns the result as a ``CreateMessageResult``.
    """
    llm = gateway or get_gateway()

    async def handle(params: types.CreateMessageRequestParams) -> types.CreateMessageResult:
        hints = (
            [h.name for h in params.modelPreferences.hints or []]
            if params.modelPreferences
            else []
        )
        hint = hints[0] if hints else None

        text_parts = [
            message.content.text
            for message in params.messages
            if isinstance(message.content, types.TextContent)
        ]
        prompt = "\n\n".join(text_parts)

        completion = await llm.complete(
            prompt,
            system=params.systemPrompt,
            model_hint=hint,
            max_tokens=params.maxTokens,
            temperature=params.temperature if params.temperature is not None else 0.0,
        )

        _log.info(
            "sampling request served",
            extra={
                "server_hint": hint,
                "client_model": completion.model,
                "provenance": completion.provenance,
                "chars": len(completion.text),
            },
        )
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=completion.text),
            model=completion.model,
            stopReason="endTurn",
        )

    return handle


#: Fields whose free text is worth translating. Identifiers, dates, drug names,
#: strengths and amounts are deliberately excluded: translating them would risk
#: corrupting clinical facts the validator compares against the EHR.
TRANSLATABLE_FIELDS = (
    ("discharge_report", "discharge_instructions"),
    ("discharge_report", "gender"),
    ("discharge_report", "service_line"),
)

TRANSLATABLE_LISTS = (
    ("discharge_report", "discharge_diagnosis"),
    ("discharge_report", "adr_allergy_info"),
    ("discharge_report", "follow_up_appointments"),
)


def build_graph(tools: ToolGateway):
    async def detect(state: NormalizerState) -> dict[str, Any]:
        text = _representative_text(state.get("record") or {})
        result = await tools.call_tool("detect_clinical_language", {"text": text})
        _log.info(
            "language detected",
            extra={
                "case_id": state.get("case_id"),
                "language": result["language"],
                "confidence": result["confidence"],
            },
        )
        return {
            "detected_language": result["language"],
            "source_language": state.get("source_language") or result["language"],
            "detection_confidence": result["confidence"],
        }

    async def translate(state: NormalizerState) -> dict[str, Any]:
        """Translate each free-text field through the Lang Bridge tool."""
        record = state.get("record") or {}
        language = state["source_language"]

        detection = state.get("detection_confidence")

        translated: dict[str, str] = {}
        confidences: dict[str, float] = {}
        samples: list[tuple[float, int]] = []
        abbreviations: dict[str, str] = {}
        model_used: str | None = None
        sampling_used = False
        warnings: list[str] = []

        async def bridge_field(key: str, value: str) -> str:
            nonlocal model_used, sampling_used
            outcome = await _bridge(tools, value, language, detection, warnings)
            if outcome is None:
                return value
            confidences[key] = outcome["translation_confidence"]
            samples.append((outcome["translation_confidence"], len(value)))
            abbreviations.update(outcome.get("abbreviations_expanded") or {})
            model_used = outcome.get("model_used") or model_used
            sampling_used = sampling_used or outcome.get("sampling_used", False)
            return outcome["translated_text"]

        for section, field in TRANSLATABLE_FIELDS:
            value = (record.get(section) or {}).get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            key = f"{section}.{field}"
            record[section][field] = await bridge_field(key, value)
            translated[key] = record[section][field]

        for section, field in TRANSLATABLE_LISTS:
            values = (record.get(section) or {}).get(field) or []
            if not values:
                continue
            record[section][field] = [
                await bridge_field(f"{section}.{field}[{index}]", str(entry))
                for index, entry in enumerate(values)
            ]

        return {
            "record": record,
            "translated": translated,
            "per_document_confidence": confidences,
            "confidence_samples": samples,
            "abbreviations": abbreviations,
            "model_used": model_used,
            "sampling_used": sampling_used,
            "warnings": warnings,
        }

    async def expand_prescriptions(state: NormalizerState) -> dict[str, Any]:
        """Expand frequency and route abbreviations on each prescription row.

        Drug names, strengths and quantities are left untouched so medication
        reconciliation still compares like with like against the EHR.
        """
        record = state.get("record") or {}
        mapping = (
            get_settings().rules.get("normalization_standards", {}).get("abbreviation_map", {})
        )
        expanded = dict(state.get("abbreviations") or {})

        for prescription in (record.get("discharge_report") or {}).get("medications") or []:
            for field in ("frequency", "route", "remarks", "dosage"):
                value = prescription.get(field)
                if not isinstance(value, str):
                    continue
                for abbreviation, expansion in mapping.items():
                    if value.strip() == abbreviation:
                        prescription[f"{field}_expanded"] = expansion
                        expanded[abbreviation] = expansion
        return {"record": record, "abbreviations": expanded}

    async def score(state: NormalizerState) -> dict[str, Any]:
        """Combine per-field confidences into one case-level score.

        Fields are weighted by source length, so a long instruction block
        counts for more than a one-word gender value. An unweighted minimum
        would let the shortest field decide the whole case.
        """
        samples = state.get("confidence_samples") or []
        threshold = get_settings().rules["quality_thresholds"]["translation_confidence_min"]

        if state["source_language"] == "en":
            # Nothing was translated, so there is no translation risk to score.
            overall = 1.0
        elif samples:
            total_weight = sum(weight for _, weight in samples) or 1
            overall = round(
                sum(conf * weight for conf, weight in samples) / total_weight, 3
            )
        else:
            overall = round(state.get("detection_confidence", 0.0) * 0.5, 3)

        warnings: list[str] = []
        if overall < threshold:
            warnings.append(
                f"Translation confidence {overall} is below the {threshold} threshold; "
                "human review is required."
            )

        _log.info(
            "normalization scored",
            extra={
                "case_id": state.get("case_id"),
                "language": state["source_language"],
                "confidence": overall,
                "below_threshold": overall < threshold,
            },
        )
        return {"translation_confidence": overall, "warnings": warnings}

    graph = StateGraph(NormalizerState)
    graph.add_node("detect", detect)
    graph.add_node("translate", translate)
    graph.add_node("expand", expand_prescriptions)
    graph.add_node("score", score)

    graph.set_entry_point("detect")

    def needs_translation(state: NormalizerState) -> str:
        return "expand" if state["source_language"] == "en" else "translate"

    graph.add_conditional_edges(
        "detect", needs_translation, {"translate": "translate", "expand": "expand"}
    )
    graph.add_edge("translate", "expand")
    graph.add_edge("expand", "score")
    graph.add_edge("score", END)

    return graph.compile(checkpointer=MemorySaver())


async def _bridge(
    tools: ToolGateway,
    text: str,
    language: str,
    detection_confidence: float | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    try:
        return await tools.call_tool(
            "medical_lang_bridge",
            {
                "text": text,
                "source_language": language,
                "detection_confidence": detection_confidence,
            },
        )
    except Exception as exc:  # noqa: BLE001 - degrade to untranslated, flagged
        _log.error("language bridge failed", extra={"error": str(exc)})
        warnings.append(f"Translation failed for one field: {exc}")
        return None


def _representative_text(record: dict[str, Any]) -> str:
    """A sample of free text wide enough to identify the language."""
    discharge = record.get("discharge_report") or {}
    parts: list[str] = []
    for value in (
        discharge.get("discharge_instructions"),
        discharge.get("service_line"),
        *(discharge.get("discharge_diagnosis") or []),
        *(discharge.get("follow_up_appointments") or []),
        *(discharge.get("adr_allergy_info") or []),
    ):
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


class ClinicalNormalizerAgent:
    def __init__(self, tools: ToolGateway) -> None:
        self.tools = tools
        self.graph = build_graph(tools)

    async def normalize(
        self, record: dict[str, Any], case_id: str, trace_id: str = ""
    ) -> dict[str, Any]:
        state: NormalizerState = {
            "case_id": case_id,
            "patient_id": record.get("patient_id", ""),
            "trace_id": trace_id,
            "record": record,
            "warnings": [],
        }
        final = await self.graph.ainvoke(
            state, config={"configurable": {"thread_id": f"{case_id}:normalize"}}
        )

        normalized = final.get("record") or record
        normalized["source_language"] = final.get("source_language", "en")

        return {
            "ok": True,
            "case_id": case_id,
            "record": normalized,
            "source_language": final.get("source_language"),
            "detected_language": final.get("detected_language"),
            "translation_confidence": final.get("translation_confidence", 0.0),
            "per_document_confidence": final.get("per_document_confidence", {}),
            "abbreviations_expanded": final.get("abbreviations", {}),
            "model_used": final.get("model_used"),
            "sampling_used": final.get("sampling_used", False),
            "warnings": final.get("warnings", []),
        }

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("_metadata") or {}
        record = payload.get("record")
        if not record:
            return {"ok": False, "error": "record is required"}
        return await self.normalize(
            record=record,
            case_id=payload.get("case_id") or metadata.get("case_id", "CASE-UNKNOWN"),
            trace_id=payload.get("trace_id") or metadata.get("trace_id", ""),
        )
