"""Host Orchestrator — Google ADK, port 8083 (doc §3, Figure 1).

Owns the discharge workflow end to end:

1. mints the ``case_id`` and the single ``trace_id`` every agent shares
2. groups incoming files by patient through the Monitor
3. drives Extractor, Normalizer and Validator over A2A
4. generates the audit report
5. indexes the case for RAG — for **every** case, including blocked ones
6. applies the HITL-2 escalation gate
7. streams the summary only when the gate says the case is clear

The gate is the safety-critical part. Doc Table 12 requires mandatory human
review when ``risk_level`` is High or ``discharge_blocked`` is true, and the
Summary Generator independently refuses blocked cases, so a blocked discharge
cannot produce a patient-facing summary even if the orchestrator were wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from hospital_ai.agents.adk.monitor import DischargeMonitorAgent
from hospital_ai.agents.adk.summary import SummaryGeneratorAgent
from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.agents.langgraph.extractor import ClinicalExtractorAgent
from hospital_ai.agents.langgraph.normalizer import ClinicalNormalizerAgent
from hospital_ai.agents.langgraph.validator import ClinicalValidationAgent
from hospital_ai.core.config import get_settings
from hospital_ai.core.ids import new_case_id, new_trace_id, utc_now_iso
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import CaseStatus, ValidationResult
from hospital_ai.guardrails import GuardrailManager
from hospital_ai.observability import Tracer
from hospital_ai.storage import CaseStore, get_store

_log = get_logger(__name__, agent="host-orchestrator")

ADK_INSTRUCTION = """\
You coordinate the hospital discharge review workflow. For each patient you
sequence extraction, language normalization, clinical validation, audit
reporting and retrieval indexing, then decide whether the case may be released
automatically or must go to a human reviewer. You never release a discharge
that is blocked or scored High risk.
"""


def build_adk_agent():
    try:
        from google.adk.agents import LlmAgent

        settings = get_settings()
        agent = LlmAgent(
            name="discharge_host_orchestrator",
            description="Coordinates the LangGraph, ADK and Agno agents over A2A.",
            instruction=ADK_INSTRUCTION,
            model=settings.llm.primary_model.replace("bedrock/", ""),
        )
        _log.info("ADK host agent constructed")
        return agent
    except Exception as exc:  # noqa: BLE001 - orchestration works without the LLM shell
        _log.warning("ADK LlmAgent unavailable", extra={"error": str(exc)})
        return None


@dataclass
class CaseOutcome:
    case_id: str
    patient_id: str
    trace_id: str
    status: CaseStatus
    validation: ValidationResult | None = None
    record: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    translation_confidence: float | None = None
    requires_hitl: bool = False
    indexed_chunks: int = 0
    guardrails: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    trace_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "trace_id": self.trace_id,
            "trace_url": self.trace_url,
            "status": self.status.value,
            "requires_hitl": self.requires_hitl,
            "risk_level": self.validation.risk_level.value if self.validation else None,
            "risk_score": self.validation.risk_score if self.validation else None,
            "discharge_blocked": self.validation.discharge_blocked if self.validation else None,
            "recommendation": self.validation.recommendation.value if self.validation else None,
            "completeness_score": self.validation.completeness_score if self.validation else None,
            "translation_confidence": self.translation_confidence,
            "findings": [f.model_dump(mode="json") for f in self.validation.findings]
            if self.validation
            else [],
            "report": self.report,
            "indexed_chunks": self.indexed_chunks,
            "guardrails": self.guardrails,
            "errors": self.errors,
        }


class HostOrchestrator:
    def __init__(
        self,
        tools: ToolGateway,
        *,
        store: CaseStore | None = None,
        rag: ClinicalRAGAgent | None = None,
    ) -> None:
        self.tools = tools
        self.store = store or get_store()
        self.monitor = DischargeMonitorAgent(tools)
        self.extractor = ClinicalExtractorAgent(tools)
        self.normalizer = ClinicalNormalizerAgent(tools)
        self.validator = ClinicalValidationAgent(tools)
        self.summary = SummaryGeneratorAgent(tools)
        self.rag = rag or ClinicalRAGAgent(tools)
        self.guardrails = GuardrailManager()
        self.adk_agent = build_adk_agent()

    # --- discovery -----------------------------------------------------------

    async def discover(self, only_new: bool = False) -> dict[str, Any]:
        return await self.monitor.scan(only_new=only_new)

    # --- the workflow --------------------------------------------------------

    async def process_patient(
        self, patient_id: str, *, case_id: str | None = None, run_no: int = 1
    ) -> CaseOutcome:
        case_id = case_id or new_case_id(patient_id)
        trace_id = new_trace_id()
        tracer = Tracer(trace_id, case_id=case_id, patient_id=patient_id)

        outcome = CaseOutcome(
            case_id=case_id,
            patient_id=patient_id,
            trace_id=trace_id,
            status=CaseStatus.CREATED,
            trace_url=tracer.url,
        )
        self.store.create_case(case_id, patient_id, trace_id)
        _log.info("case started", extra={"case_id": case_id, "patient_id": patient_id})

        try:
            # 2. Extraction
            with tracer.agent_span("clinical-extractor", {"patient_id": patient_id}) as span:
                extracted = await self.extractor.extract(patient_id, case_id, trace_id)
                span.output = {"ok": extracted["ok"], "warnings": extracted["warnings"]}
            if not extracted["ok"]:
                outcome.errors.extend(extracted["errors"])
                outcome.status = CaseStatus.FAILED
                self.store.update_status(case_id, CaseStatus.FAILED)
                tracer.finish(outcome.to_dict())
                return outcome

            record = extracted["record"]
            outcome.status = CaseStatus.EXTRACTED
            self.store.update_status(case_id, CaseStatus.EXTRACTED)

            # 3. Normalization
            with tracer.agent_span("clinical-normalizer", {"case_id": case_id}) as span:
                normalized = await self.normalizer.normalize(record, case_id, trace_id)
                span.output = {
                    "language": normalized["source_language"],
                    "confidence": normalized["translation_confidence"],
                }
            if normalized.get("sampling_used"):
                tracer.log_sampling(
                    server_hints=["nova-lite"],
                    client_model=normalized.get("model_used") or "unknown",
                    result_preview=str(normalized["record"].get("discharge_report", {}).get("discharge_instructions", ""))[:500],
                )

            record = normalized["record"]
            outcome.record = record
            outcome.translation_confidence = normalized["translation_confidence"]
            outcome.status = CaseStatus.NORMALIZED
            # Persist the normalized record here: the summary stream and any
            # later HITL correction both read it back from the store.
            self.store.update_status(case_id, CaseStatus.NORMALIZED, record=record)

            # 4. Validation (HITL-1 elicitation happens inside)
            with tracer.agent_span("clinical-validator", {"case_id": case_id}) as span:
                validation = await self.validator.validate(
                    record,
                    case_id,
                    trace_id=trace_id,
                    run_no=run_no,
                    translation_confidence=outcome.translation_confidence,
                )
                span.output = {
                    "risk_level": validation.risk_level.value,
                    "blocked": validation.discharge_blocked,
                    "findings": len(validation.findings),
                }
            for elicitation in validation.elicitations:
                tracer.log_elicitation(
                    elicitation.fields_requested,
                    elicitation.requested_schema,
                    elicitation.action.value,
                    elicitation.response,
                )

            outcome.validation = validation
            outcome.status = CaseStatus.VALIDATED
            self.store.save_validation(case_id, validation)

            # 5. Audit report
            with tracer.tool_span("clinical_insight_reporter", {"case_id": case_id}) as span:
                report = await self.tools.call_tool(
                    "clinical_insight_reporter",
                    {
                        "case_id": case_id,
                        "patient_id": patient_id,
                        "trace_id": trace_id,
                        "validation": validation.model_dump(mode="json"),
                        "patient_name": (record.get("discharge_report") or {}).get("patient_name"),
                        "bill": record.get("bill"),
                        "translation_confidence": outcome.translation_confidence,
                        "elicitations": [e.model_dump(mode="json") for e in validation.elicitations],
                        "service_line": validation.service_line,
                    },
                )
                span.output = report["artifacts"]
            outcome.report = report
            outcome.status = CaseStatus.REPORTED
            self.store.update_status(case_id, CaseStatus.REPORTED)

            # 6. RAG indexing — runs for every case, blocked ones included, so
            #    staff can query a case while it waits in HITL.
            with tracer.agent_span("agno-rag-indexing", {"case_id": case_id}) as span:
                indexed = self.rag.index_case(case_id, record)
                span.output = {"chunks": indexed["chunks_indexed"]}
            outcome.indexed_chunks = indexed["chunks_indexed"]
            outcome.status = CaseStatus.INDEXED

            # 7. HITL-2 escalation gate
            escalation = self.guardrails.check_hitl_escalation(
                validation.risk_level.value, validation.discharge_blocked
            )
            tracer.log_guardrail("GuardrailManager", escalation.blocked, escalation.detail)
            outcome.requires_hitl = escalation.blocked
            outcome.guardrails = self.guardrails.summary()

            outcome.status = (
                CaseStatus.HITL_PENDING if escalation.blocked else CaseStatus.SUMMARY_READY
            )
            self.store.update_status(case_id, outcome.status, validation=validation)

            _log.info(
                "case processed",
                extra={
                    "case_id": case_id,
                    "risk_level": validation.risk_level.value,
                    "blocked": validation.discharge_blocked,
                    "requires_hitl": outcome.requires_hitl,
                },
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced on the case
            tracer.log_error("host-orchestrator", exc, fallback="case marked FAILED")
            outcome.errors.append(f"{type(exc).__name__}: {exc}")
            outcome.status = CaseStatus.FAILED
            self.store.update_status(case_id, CaseStatus.FAILED)
            _log.error("case failed", extra={"case_id": case_id}, exc_info=True)

        tracer.finish(outcome.to_dict())
        return outcome

    async def process_all(self, only_new: bool = False) -> list[CaseOutcome]:
        discovered = await self.discover(only_new=only_new)
        outcomes = []
        for entry in discovered["patients"]:
            outcomes.append(await self.process_patient(entry["patient_id"]))
        self.monitor.acknowledge(discovered["patients"])
        return outcomes

    # --- summary -------------------------------------------------------------

    async def stream_summary(self, case_id: str) -> AsyncIterator[dict[str, Any]]:
        """Step 7: only reached when the escalation gate cleared the case."""
        case = self.store.get_case(case_id)
        if case is None:
            yield {"type": "error", "error": f"Unknown case {case_id}"}
            return

        record = self.store.get_record(case_id)
        if record is None:
            yield {"type": "error", "error": f"No clinical record stored for {case_id}"}
            return

        async for event in self.summary.stream_summary(
            record,
            case_id=case_id,
            risk_level=case.get("risk_level") or "Low",
            discharge_blocked=bool(case.get("discharge_blocked")),
        ):
            if event["type"] == "complete":
                self.store.save_summary(case_id, event["summary"])
                self.store.update_status(case_id, CaseStatus.CLOSED)
            yield event

    # --- HITL re-run ---------------------------------------------------------

    async def revalidate(
        self, case_id: str, corrections: dict[str, Any] | None = None
    ) -> CaseOutcome:
        """Re-run validation after a human correction (HITL-2 -> step 4)."""
        case = self.store.get_case(case_id)
        record = self.store.get_record(case_id)
        if case is None or record is None:
            raise KeyError(f"Unknown case {case_id}")

        if corrections:
            record = self.store.apply_corrections(case_id, corrections)

        run_no = self.store.next_run_no(case_id)
        trace_id = case["trace_id"]
        tracer = Tracer(trace_id, case_id=case_id, patient_id=case["patient_id"])

        with tracer.agent_span("clinical-validator-rerun", {"run_no": run_no}) as span:
            validation = await self.validator.validate(
                record, case_id, trace_id=trace_id, run_no=run_no
            )
            span.output = {
                "risk_level": validation.risk_level.value,
                "blocked": validation.discharge_blocked,
            }

        escalation = self.guardrails.check_hitl_escalation(
            validation.risk_level.value, validation.discharge_blocked
        )
        status = CaseStatus.HITL_PENDING if escalation.blocked else CaseStatus.SUMMARY_READY
        self.store.save_validation(case_id, validation)
        self.store.update_status(case_id, status, validation=validation)

        # Re-index so the RAG answers reflect the corrected record.
        self.rag.index_case(case_id, record)
        tracer.finish({"run_no": run_no, "status": status.value})

        return CaseOutcome(
            case_id=case_id,
            patient_id=case["patient_id"],
            trace_id=trace_id,
            status=status,
            validation=validation,
            record=record,
            requires_hitl=escalation.blocked,
            trace_url=tracer.url,
        )

    # --- A2A entry point -----------------------------------------------------

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action", "process")

        if action == "discover":
            return await self.discover(only_new=bool(payload.get("only_new")))
        if action == "revalidate":
            outcome = await self.revalidate(payload["case_id"], payload.get("corrections"))
            return outcome.to_dict()
        if action == "process_all":
            return {
                "ok": True,
                "processed_at": utc_now_iso(),
                "cases": [o.to_dict() for o in await self.process_all()],
            }

        patient_id = payload.get("patient_id")
        if not patient_id:
            return {"ok": False, "error": "patient_id is required"}
        outcome = await self.process_patient(patient_id, case_id=payload.get("case_id"))
        return outcome.to_dict()
