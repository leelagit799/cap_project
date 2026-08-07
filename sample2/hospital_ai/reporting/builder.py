"""Audit report generation — doc §2.5.

Produces the three artifacts the specification requires for every case:
JSON for system consumption, HTML for clinicians, and PDF for the record.

Every report is stamped with ``rules_version`` (the SHA-256 of ``rules.yaml``)
so an audit can be reproduced against the exact policy that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from hospital_ai.core.config import get_settings
from hospital_ai.core.ids import utc_now_iso
from hospital_ai.core.logging import get_logger
from hospital_ai.observability import trace_url

_log = get_logger(__name__, component="reporter")

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass
class ReportArtifacts:
    json_path: Path
    html_path: Path
    pdf_path: Path | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "json": str(self.json_path),
            "html": str(self.html_path),
            "pdf": str(self.pdf_path) if self.pdf_path else None,
            "pdf_generated": self.pdf_path is not None,
        }


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def build_payload(
    *,
    case_id: str,
    patient_id: str,
    trace_id: str,
    validation: dict[str, Any],
    patient_name: str | None = None,
    bill: dict[str, Any] | None = None,
    translation_confidence: float | None = None,
    audit_trail: list[dict[str, Any]] | None = None,
    elicitations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical report payload.

    Doc §2.5 requires: missing fields, EHR discrepancies, medication conflicts,
    translation confidence, risk level, recommendation, audit trail with
    LangFuse trace IDs, and the bill amount plus payment status.
    """
    settings = get_settings()
    findings = validation.get("findings", [])

    langfuse_url = trace_url(trace_id)

    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "trace_id": trace_id,
        "langfuse_url": langfuse_url,
        "generated_at": utc_now_iso(),
        "rules_version": validation.get("rules_version") or settings.rules_version,
        "risk_score": validation.get("risk_score", 0),
        "risk_level": validation.get("risk_level", "Low"),
        "recommendation": validation.get("recommendation", "Approve"),
        "recommendation_text": validation.get("recommendation_text", ""),
        "discharge_blocked": validation.get("discharge_blocked", False),
        "completeness_score": validation.get("completeness_score", 0.0),
        "translation_confidence": translation_confidence,
        "triggered_guardrails": validation.get("triggered_guardrails", []),
        "findings": findings,
        "missing_fields": [
            f["field"] for f in findings
            if str(f.get("rule_id", "")).startswith("missing_field.") and f.get("field")
        ],
        "ehr_discrepancies": [
            f for f in findings
            if f.get("rule_id") in {
                "med_omission_check", "diagnosis_mismatch_check",
                "lab_follow_up_mismatch_check", "follow_up_missing_check",
            }
        ],
        "medication_conflicts": [
            f for f in findings
            if f.get("rule_id") in {
                "allergy_contradiction_check", "high_risk_med_missing_in_ehr",
                "incomplete_prescription_fields",
            }
        ],
        "bill_total": (bill or {}).get("total_amount"),
        "currency": (bill or {}).get("currency") or "",
        "payment_status": (bill or {}).get("payment_status") or "UNKNOWN",
        "elicitations": elicitations or [],
        "audit_trail": audit_trail or [],
    }


def render_html(payload: dict[str, Any]) -> str:
    return _environment().get_template("audit_report.html.j2").render(**payload)


def render_pdf(html: str, target: Path) -> Path | None:
    """Render the HTML report to PDF.

    WeasyPrint needs system Pango/Cairo libraries. When they are absent the
    JSON and HTML artifacts are still produced and the caller is told the PDF
    was skipped, rather than failing the whole case over a packaging detail.
    """
    try:
        from weasyprint import HTML
    except Exception as exc:  # noqa: BLE001 - import-time system library failure
        _log.warning("PDF rendering unavailable", extra={"error": str(exc)})
        return None

    try:
        HTML(string=html).write_pdf(str(target))
        return target
    except Exception as exc:  # noqa: BLE001 - rendering must not fail the case
        _log.warning("PDF rendering failed", extra={"error": str(exc)})
        return None


def generate(
    *,
    case_id: str,
    patient_id: str,
    trace_id: str,
    validation: dict[str, Any],
    patient_name: str | None = None,
    bill: dict[str, Any] | None = None,
    translation_confidence: float | None = None,
    audit_trail: list[dict[str, Any]] | None = None,
    elicitations: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("json", "html", "pdf"),
) -> ReportArtifacts:
    """Write the audit report in every requested format."""
    settings = get_settings()
    directory = output_dir or (settings.reports_dir / case_id)
    directory.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        case_id=case_id,
        patient_id=patient_id,
        trace_id=trace_id,
        validation=validation,
        patient_name=patient_name,
        bill=bill,
        translation_confidence=translation_confidence,
        audit_trail=audit_trail,
        elicitations=elicitations,
    )

    json_path = directory / "audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    html = render_html(payload)
    html_path = directory / "audit.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path = None
    if "pdf" in formats:
        pdf_path = render_pdf(html, directory / "audit.pdf")

    _log.info(
        "audit report generated",
        extra={
            "case_id": case_id,
            "risk_level": payload["risk_level"],
            "blocked": payload["discharge_blocked"],
            "pdf": pdf_path is not None,
        },
    )
    return ReportArtifacts(
        json_path=json_path, html_path=html_path, pdf_path=pdf_path, payload=payload
    )
