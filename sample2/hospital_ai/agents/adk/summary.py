"""Discharge Summary Generator — Google ADK, port 8104, STREAMING (doc §2.5, Table 10).

Streams a patient-friendly summary section by section in the order the
specification fixes: patient, medications, labs, bill, instructions.

Two guarantees this agent enforces:

- it refuses to generate for a blocked discharge, returning a typed
  ``SummaryBlocked`` artifact instead (doc Table 12: "Never allow blocked
  discharge summaries")
- every section passes the toxicity and unsafe-advice filter before it is
  emitted, and a section that trips it is regenerated once from the record
  rather than shipped
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import DischargeSummary, SummarySection
from hospital_ai.guardrails import GuardrailManager
from hospital_ai.llm.gateway import LLMGateway, get_gateway

_log = get_logger(__name__, agent="summary-generator")

ADK_INSTRUCTION = """\
You write discharge summaries that patients and their families can actually
follow. Use plain English, short sentences and a calm tone. State only what the
clinical record contains: never invent a dose, a date, a diagnosis or a
reassurance. You are not the treating clinician and must not give new medical
advice.
"""

#: Doc Table 10 fixes this order for progressive delivery.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("patient", "About your stay"),
    ("medications", "Your medications"),
    ("labs", "Your test results"),
    ("bill", "Your hospital bill"),
    ("instructions", "What to do next"),
)


def build_adk_agent():
    try:
        from google.adk.agents import LlmAgent

        settings = get_settings()
        agent = LlmAgent(
            name="discharge_summary_generator",
            description="Writes patient-friendly discharge summaries, streamed section by section.",
            instruction=ADK_INSTRUCTION,
            model=settings.llm.primary_model.replace("bedrock/", ""),
        )
        _log.info("ADK summary agent constructed")
        return agent
    except Exception as exc:  # noqa: BLE001 - generation works without the LLM shell
        _log.warning("ADK LlmAgent unavailable", extra={"error": str(exc)})
        return None


def _section_facts(name: str, record: dict[str, Any]) -> str:
    """The record excerpt a section is allowed to draw on.

    Each section sees only its own facts, so the model cannot pull a lab value
    into the billing paragraph or invent a medication from context it should
    not have had.
    """
    discharge = record.get("discharge_report") or {}
    labs = record.get("lab_report") or {}
    bill = record.get("bill") or {}

    if name == "patient":
        return (
            f"Name: {discharge.get('patient_name')}\n"
            f"Admitted: {discharge.get('admission_date')}\n"
            f"Discharged: {discharge.get('discharge_date')}\n"
            f"Ward {discharge.get('ward')}, bed {discharge.get('bed_no')}\n"
            f"Diagnosis: {'; '.join(str(d) for d in discharge.get('discharge_diagnosis') or []) or 'not recorded'}\n"
            f"Attending physician: {discharge.get('attending_physician')}\n"
            f"Known allergies: {'; '.join(str(a) for a in discharge.get('adr_allergy_info') or []) or 'none recorded'}"
        )

    if name == "medications":
        rows = [
            f"- {m.get('medicine_name')} {m.get('strength')}: take {m.get('dosage')}, "
            f"{m.get('frequency_expanded') or m.get('frequency')}, by {m.get('route')}, "
            f"for {m.get('period')}. {m.get('remarks') or ''}".strip()
            for m in discharge.get("medications") or []
        ]
        return "\n".join(rows) or "No discharge medications were recorded."

    if name == "labs":
        rows = [
            f"- {t.get('test')}: {t.get('value')} {t.get('unit') or ''} "
            f"(normal range {t.get('reference_range')}) — "
            f"{'outside the normal range' if t.get('abnormal') else 'normal'}"
            + (f". Plan: {t['documented_action']}" if t.get("documented_action") else "")
            for t in labs.get("tests") or []
        ]
        return "\n".join(rows) or "No laboratory results were recorded."

    if name == "bill":
        return (
            f"Total: {bill.get('currency') or ''} {bill.get('total_amount')}\n"
            f"Payment status: {bill.get('payment_status')}\n"
            f"Payment method: {bill.get('payment_method') or 'not recorded'}"
        )

    follow_ups = discharge.get("follow_up_appointments") or []
    return (
        f"Follow-up appointments: {'; '.join(str(f) for f in follow_ups) or 'none scheduled'}\n"
        f"Instructions from your care team: {discharge.get('discharge_instructions') or 'none recorded'}"
    )


class SummaryGeneratorAgent:
    def __init__(
        self,
        tools: ToolGateway,
        llm: LLMGateway | None = None,
        guardrails: GuardrailManager | None = None,
    ) -> None:
        self.tools = tools
        self.llm = llm or get_gateway()
        self.guardrails = guardrails or GuardrailManager()
        self.adk_agent = build_adk_agent()

    async def _prompt(self, risk_level: str, audience: str) -> str:
        # Fetched via MCP Prompts, never hardcoded (doc Table 6).
        return await self.tools.get_prompt(
            "summary-generation-prompt", {"risk_level": risk_level, "audience": audience}
        )

    async def _write_section(
        self, name: str, title: str, facts: str, system: str, patient_name: str
    ) -> str:
        instruction = (
            f"Write only the '{title}' section of a discharge summary for "
            f"{patient_name}. Use two to four short sentences of plain English. "
            f"Use only these facts and add nothing else:\n\n{facts}"
        )
        completion = await self.llm.complete(instruction, system=system, temperature=0.2)
        text = completion.text.strip()

        verdict = self.guardrails.check_toxicity(text)
        if verdict.blocked:
            _log.warning(
                "section regenerated after guardrail block",
                extra={"section": name, "detail": verdict.detail},
            )
            retry = await self.llm.complete(
                instruction
                + "\n\nDo not give any medical advice beyond what the facts state.",
                system=system,
                temperature=0.0,
            )
            text = retry.text.strip()
            if self.guardrails.check_toxicity(text).blocked:
                # Two failures means fall back to the facts themselves, which
                # are by definition what the record says.
                return facts
        return text

    async def stream_summary(
        self,
        record: dict[str, Any],
        *,
        case_id: str,
        risk_level: str = "Low",
        discharge_blocked: bool = False,
        audience: str = "patient",
    ) -> AsyncIterator[dict[str, Any]]:
        patient_id = record.get("patient_id", "unknown")

        escalation = self.guardrails.check_hitl_escalation(risk_level, discharge_blocked)
        if escalation.blocked:
            _log.warning(
                "summary generation refused for a blocked discharge",
                extra={"case_id": case_id, "risk_level": risk_level},
            )
            yield {
                "type": "blocked",
                "case_id": case_id,
                "patient_id": patient_id,
                "risk_level": risk_level,
                "reason": escalation.detail,
                "message": (
                    "This discharge is blocked pending clinician review, so no "
                    "patient-facing summary was generated."
                ),
            }
            return

        discharge = record.get("discharge_report") or {}
        patient_name = discharge.get("patient_name") or patient_id
        system = await self._prompt(risk_level, audience)

        yield {
            "type": "start",
            "case_id": case_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "sections": [name for name, _ in SECTIONS],
        }

        written: list[SummarySection] = []
        for order, (name, title) in enumerate(SECTIONS, start=1):
            facts = _section_facts(name, record)
            content = await self._write_section(name, title, facts, system, patient_name)
            section = SummarySection(name=name, title=title, content=content, order=order)
            written.append(section)
            yield {"type": "section", **section.model_dump()}

        summary = DischargeSummary(
            case_id=case_id,
            patient_id=patient_id,
            sections=written,
            model_used=self.llm.primary,
        )
        yield {"type": "complete", "summary": summary.model_dump(mode="json")}

    async def generate(
        self,
        record: dict[str, Any],
        *,
        case_id: str,
        risk_level: str = "Low",
        discharge_blocked: bool = False,
        audience: str = "patient",
    ) -> dict[str, Any]:
        """Non-streaming convenience path, used by the dashboard and reports."""
        sections: list[dict[str, Any]] = []
        blocked: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None

        async for event in self.stream_summary(
            record,
            case_id=case_id,
            risk_level=risk_level,
            discharge_blocked=discharge_blocked,
            audience=audience,
        ):
            if event["type"] == "section":
                sections.append(event)
            elif event["type"] == "blocked":
                blocked = event
            elif event["type"] == "complete":
                summary = event["summary"]

        if blocked is not None:
            return {"ok": False, "blocked": True, **blocked}
        return {"ok": True, "blocked": False, "summary": summary, "sections": sections}

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("_metadata") or {}
        record = payload.get("record")
        if not record:
            return {"ok": False, "error": "record is required"}
        return await self.generate(
            record,
            case_id=payload.get("case_id") or metadata.get("case_id", "CASE-UNKNOWN"),
            risk_level=payload.get("risk_level", "Low"),
            discharge_blocked=bool(payload.get("discharge_blocked", False)),
            audience=payload.get("audience", "patient"),
        )

    async def handle_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        metadata = payload.get("_metadata") or {}
        record = payload.get("record")
        if not record:
            yield {"type": "error", "error": "record is required"}
            return
        async for event in self.stream_summary(
            record,
            case_id=payload.get("case_id") or metadata.get("case_id", "CASE-UNKNOWN"),
            risk_level=payload.get("risk_level", "Low"),
            discharge_blocked=bool(payload.get("discharge_blocked", False)),
            audience=payload.get("audience", "patient"),
        ):
            yield event
