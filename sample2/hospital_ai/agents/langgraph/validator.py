"""Clinical Validation Agent — LangGraph, port 8101 (doc §2.4).

A ``StateGraph`` that loads its rules as an **MCP Resource**, checks
completeness (Table 3), cross-validates against the Mock EHR (Table 4), scores
risk through the **Secondary MCP server**, and decides whether discharge is
blocked.

Graph shape::

    load_rules -> completeness -> cross_validate -> score -> decide

``completeness`` runs the Clinical Rules Engine Tool, which may raise an **MCP
Elicitation** for non-blocking gaps in the same pass — HITL-1 in the workflow.
The graph supports re-runs: HITL-2 corrections come back in as a new run with
an incremented ``run_no``, which is why validation history is keyed per run
rather than overwritten.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import (
    ElicitationRecord,
    Finding,
    Recommendation,
    RiskLevel,
    Severity,
    ValidationResult,
)

_log = get_logger(__name__, agent="clinical-validator")


def _merge(left: list, right: list) -> list:
    return [*left, *right]


class ValidatorState(TypedDict, total=False):
    case_id: str
    patient_id: str
    trace_id: str
    run_no: int
    record: dict[str, Any]
    packet: dict[str, Any]
    translation_confidence: float | None
    service_line: str | None
    rules_text: str
    rules_version: str
    completeness_score: float
    elicitation: dict[str, Any] | None
    risk: dict[str, Any]
    findings: Annotated[list[dict[str, Any]], _merge]
    errors: Annotated[list[str], _merge]


def _packet_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Shape a ClinicalRecord into the {doc_type: document} the tools expect."""
    discharge = dict(record.get("discharge_report") or {}) or None
    if discharge is not None:
        # Table 3 names a single `doctors` field; the model splits attending
        # from consulting, so present the merged view for completeness checks.
        doctors = [discharge.get("attending_physician")] if discharge.get("attending_physician") else []
        doctors += list(discharge.get("consulting_doctors") or [])
        discharge["doctors"] = doctors

    return {
        "discharge_report": discharge,
        "lab_report": dict(record.get("lab_report") or {}) or None,
        "bill": dict(record.get("bill") or {}) or None,
    }


def build_graph(tools: ToolGateway):
    async def load_rules(state: ValidatorState) -> dict[str, Any]:
        """Fetch rules through MCP Resources rather than reading the file.

        Doc §2.4.1 requires rules.yaml to be "fetched as MCP Resource at
        runtime", which makes the rule set a negotiated contract instead of a
        local file read.
        """
        completeness = await tools.read_resource("resource://clinical-rules/completeness")
        cross = await tools.read_resource("resource://clinical-rules/cross-validation")
        version = ""
        try:
            version = (yaml.safe_load(completeness) or {}).get("rules_version", "")
        except yaml.YAMLError:
            pass

        return {
            "rules_text": completeness + "\n" + cross,
            "rules_version": version or get_settings().rules_version,
            "packet": _packet_from_record(state.get("record") or {}),
        }

    async def completeness(state: ValidatorState) -> dict[str, Any]:
        """Table 3 checks, with MCP Elicitation for non-blocking gaps (HITL-1)."""
        result = await tools.call_tool(
            "clinical_rules_engine",
            {"packet": state["packet"], "elicit": state.get("run_no", 1) == 1},
        )
        _log.info(
            "completeness checked",
            extra={
                "case_id": state.get("case_id"),
                "score": result["completeness_score"],
                "findings": len(result["findings"]),
                "elicitation": (result.get("elicitation") or {}).get("action"),
            },
        )
        return {
            "findings": result["findings"],
            "completeness_score": result["completeness_score"],
            "elicitation": result.get("elicitation"),
            # The tool may have written reviewer-supplied values into the packet.
            "packet": state["packet"],
        }

    async def cross_validate(state: ValidatorState) -> dict[str, Any]:
        """Table 4 checks against the Mock EHR."""
        try:
            result = await tools.call_tool(
                "ehr_validation",
                {"patient_id": state["patient_id"], "packet": state["packet"]},
            )
        except Exception as exc:  # noqa: BLE001 - an EHR outage must block, not pass
            _log.error("cross-validation failed", extra={"error": str(exc)})
            return {
                "findings": [
                    {
                        "rule_id": "ehr_unavailable",
                        "severity": Severity.CRITICAL.value,
                        "message": f"EHR cross-validation could not run: {exc}",
                        "weight": 8,
                        "blocking": True,
                        "resolved": False,
                    }
                ],
                "errors": [str(exc)],
            }

        service_line = (result.get("ehr_patient") or {}).get("service_line")
        return {"findings": result["findings"], "service_line": service_line}

    async def score(state: ValidatorState) -> dict[str, Any]:
        """Compute the composite risk score on the Secondary MCP server."""
        risk = await tools.call_tool(
            "calculate_risk_score",
            {
                "findings": state.get("findings", []),
                "translation_confidence": state.get("translation_confidence"),
                "service_line": state.get("service_line"),
            },
        )
        _log.info(
            "risk scored",
            extra={
                "case_id": state.get("case_id"),
                "score": risk["risk_score"],
                "level": risk["risk_level"],
                "blocked": risk["discharge_blocked"],
            },
        )
        return {"risk": risk}

    async def decide(state: ValidatorState) -> dict[str, Any]:
        return {}

    graph = StateGraph(ValidatorState)
    graph.add_node("load_rules", load_rules)
    graph.add_node("completeness", completeness)
    graph.add_node("cross_validate", cross_validate)
    graph.add_node("score", score)
    graph.add_node("decide", decide)

    graph.set_entry_point("load_rules")
    graph.add_edge("load_rules", "completeness")

    def has_packet(state: ValidatorState) -> str:
        """Skip EHR cross-validation when there is no discharge report to compare."""
        return "cross_validate" if (state.get("packet") or {}).get("discharge_report") else "score"

    graph.add_conditional_edges(
        "completeness", has_packet, {"cross_validate": "cross_validate", "score": "score"}
    )
    graph.add_edge("cross_validate", "score")
    graph.add_edge("score", "decide")
    graph.add_edge("decide", END)

    return graph.compile(checkpointer=MemorySaver())


class ClinicalValidationAgent:
    def __init__(self, tools: ToolGateway) -> None:
        self.tools = tools
        self.graph = build_graph(tools)

    async def validate(
        self,
        record: dict[str, Any],
        case_id: str,
        *,
        trace_id: str = "",
        run_no: int = 1,
        translation_confidence: float | None = None,
    ) -> ValidationResult:
        state: ValidatorState = {
            "case_id": case_id,
            "patient_id": record.get("patient_id", ""),
            "trace_id": trace_id,
            "run_no": run_no,
            "record": record,
            "translation_confidence": translation_confidence,
            "findings": [],
            "errors": [],
        }
        final = await self.graph.ainvoke(
            state, config={"configurable": {"thread_id": f"{case_id}:validate:{run_no}"}}
        )

        risk = final.get("risk") or {}
        findings = [Finding(**f) for f in final.get("findings", [])]

        elicitations: list[ElicitationRecord] = []
        raw_elicitation = final.get("elicitation")
        if raw_elicitation:
            elicitations.append(
                ElicitationRecord(
                    fields_requested=raw_elicitation.get("fields_requested", []),
                    requested_schema=raw_elicitation.get("requested_schema", {}),
                    action=raw_elicitation.get("action", "decline"),
                    response=raw_elicitation.get("response", {}),
                )
            )

        return ValidationResult(
            case_id=case_id,
            patient_id=state["patient_id"],
            run_no=run_no,
            findings=findings,
            elicitations=elicitations,
            completeness_score=final.get("completeness_score", 0.0),
            risk_score=risk.get("risk_score", 0),
            risk_level=RiskLevel(risk.get("risk_level", "Low")),
            discharge_blocked=risk.get("discharge_blocked", False),
            recommendation=Recommendation(risk.get("recommendation", "Approve")),
            recommendation_text=risk.get("recommendation_text", ""),
            rules_version=final.get("rules_version", ""),
            translation_confidence=translation_confidence,
            triggered_guardrails=risk.get("triggered_guardrails", []),
            service_line=final.get("service_line"),
        )

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("_metadata") or {}
        record = payload.get("record")
        if not record:
            return {"ok": False, "error": "record is required"}

        result = await self.validate(
            record=record,
            case_id=payload.get("case_id") or metadata.get("case_id", "CASE-UNKNOWN"),
            trace_id=payload.get("trace_id") or metadata.get("trace_id", ""),
            run_no=int(payload.get("run_no", 1)),
            translation_confidence=payload.get("translation_confidence"),
        )
        return {"ok": True, **result.model_dump(mode="json")}
