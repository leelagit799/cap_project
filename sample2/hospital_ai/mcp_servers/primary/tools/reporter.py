"""Clinical Insight Reporter Tool — Tool + Resources (doc Table 7).

Generates the discharge audit report in JSON, HTML and PDF. The HTML is
rendered from the same Jinja2 template the server publishes as
``resource://report-template/html``, so what a clinician reads and what a
client can fetch as a resource never drift apart.

Doc §2.5 lists this as report generation, and the workflow places it as an MCP
tool rather than an A2A agent.
"""

from __future__ import annotations

from typing import Any

from hospital_ai.analytics.risk import assess
from hospital_ai.core.logging import get_logger
from hospital_ai.reporting import builder

_log = get_logger(__name__, tool="clinical-insight-reporter")


def build_report(
    case_id: str,
    patient_id: str,
    trace_id: str,
    validation: dict[str, Any],
    patient_name: str | None = None,
    bill: dict[str, Any] | None = None,
    translation_confidence: float | None = None,
    audit_trail: list[dict[str, Any]] | None = None,
    elicitations: list[dict[str, Any]] | None = None,
    service_line: str | None = None,
) -> dict[str, Any]:
    """Score the case if needed, then write all three artifacts."""
    enriched = dict(validation)

    if "risk_level" not in enriched:
        assessment = assess(
            enriched.get("findings", []),
            translation_confidence=translation_confidence,
            service_line=service_line,
        )
        enriched.update(assessment.to_dict())

    artifacts = builder.generate(
        case_id=case_id,
        patient_id=patient_id,
        trace_id=trace_id,
        validation=enriched,
        patient_name=patient_name,
        bill=bill,
        translation_confidence=translation_confidence,
        audit_trail=audit_trail,
        elicitations=elicitations,
    )

    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "artifacts": artifacts.to_dict(),
        "risk_level": artifacts.payload["risk_level"],
        "risk_score": artifacts.payload["risk_score"],
        "recommendation": artifacts.payload["recommendation"],
        "discharge_blocked": artifacts.payload["discharge_blocked"],
        "rules_version": artifacts.payload["rules_version"],
        "report": artifacts.payload,
    }


def register(mcp) -> None:
    @mcp.tool(
        name="clinical_insight_reporter",
        description=(
            "Generate the discharge audit report as JSON, HTML and PDF, "
            "stamped with the rules version and LangFuse trace id."
        ),
    )
    def clinical_insight_reporter(
        case_id: str,
        patient_id: str,
        trace_id: str,
        validation: dict[str, Any],
        patient_name: str | None = None,
        bill: dict[str, Any] | None = None,
        translation_confidence: float | None = None,
        audit_trail: list[dict[str, Any]] | None = None,
        elicitations: list[dict[str, Any]] | None = None,
        service_line: str | None = None,
    ) -> dict[str, Any]:
        return build_report(
            case_id=case_id,
            patient_id=patient_id,
            trace_id=trace_id,
            validation=validation,
            patient_name=patient_name,
            bill=bill,
            translation_confidence=translation_confidence,
            audit_trail=audit_trail,
            elicitations=elicitations,
            service_line=service_line,
        )
