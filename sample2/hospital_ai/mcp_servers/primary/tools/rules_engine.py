"""Clinical Rules Engine Tool — Tool + Elicitation (doc §2.4.1, Table 9).

Validates a discharge packet for completeness against the required/blocking
matrix of doc Table 3, then — for *non-blocking* gaps only — calls
``ctx.elicit()`` with a Pydantic schema so a human reviewer can fill them in
during the same pass.

All three elicitation outcomes are handled distinctly, as the specification
requires:

- **accept**  — use the reviewer's input and continue
- **decline** — mark unresolved and flag for HITL
- **cancel**  — abort and escalate

Blocking gaps are never elicited. A missing patient_id or an unapproved
discharge is not a form-fill; it goes straight to HITL-2.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field, create_model

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import DocType, ElicitAction, Severity

_log = get_logger(__name__, tool="clinical-rules-engine")


#: Doc Table 3 — required fields per document, and which of them block release.
DOCUMENT_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    DocType.DISCHARGE_REPORT.value: {
        "required": (
            "patient_id", "patient_name", "age", "gender", "address",
            "admission_date", "discharge_date", "ward", "bed_no", "doctors",
            "discharge_diagnosis", "medications", "adr_allergy_info",
            "follow_up_appointments", "discharge_instructions",
            "discharge_approved_by", "discharge_approved",
        ),
        "blocking": (
            "patient_id", "patient_name", "discharge_diagnosis",
            "discharge_approved", "medications",
        ),
    },
    DocType.LAB_REPORT.value: {
        "required": ("patient_id", "vendor_name", "lab_name", "report_date", "tests"),
        "blocking": ("patient_id", "tests"),
    },
    DocType.BILL.value: {
        "required": (
            "patient_id", "hospital_name", "billing_date", "line_items",
            "total_amount", "payment_status",
        ),
        "blocking": ("patient_id", "total_amount", "payment_status"),
    },
    "prescription": {
        "required": (
            "sl_no", "medicine_name", "strength", "dosage", "frequency",
            "route", "period", "remarks", "total_quantity",
        ),
        "blocking": ("medicine_name", "strength", "frequency", "route"),
    },
}

#: Fields rules.yaml scores more leniently than a generic missing field.
SOFT_FIELD_WEIGHTS = {"address": "missing_address", "gender": "missing_gender"}

#: Free-text fields a reviewer can reasonably supply through an elicitation form.
ELICITABLE_FIELDS = {
    "address", "gender", "age", "ward", "bed_no", "admission_date",
    "discharge_date", "doctors", "follow_up_appointments",
    "discharge_instructions", "vendor_name", "lab_name", "report_date",
    "hospital_name", "billing_date",
}

_FIELD_TYPES: dict[str, type] = {"age": int}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def check_completeness(
    document: dict[str, Any], doc_type: str
) -> tuple[list[str], list[str]]:
    """Return ``(missing_required, missing_blocking)`` for one document."""
    spec = DOCUMENT_REQUIREMENTS.get(doc_type)
    if spec is None:
        return [], []

    missing = [field for field in spec["required"] if _is_missing(document.get(field))]
    blocking = [field for field in missing if field in spec["blocking"]]
    return missing, blocking


def check_prescriptions(medications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every prescription row must populate all nine columns."""
    spec = DOCUMENT_REQUIREMENTS["prescription"]
    issues: list[dict[str, Any]] = []

    for index, med in enumerate(medications, start=1):
        missing = [field for field in spec["required"] if _is_missing(med.get(field))]
        if not missing:
            continue
        issues.append(
            {
                "row": index,
                "medicine_name": med.get("medicine_name") or f"row {index}",
                "missing_fields": missing,
                "blocking": [f for f in missing if f in spec["blocking"]],
            }
        )
    return issues


def build_elicitation_schema(fields: list[str]) -> type[BaseModel]:
    """Build the Pydantic schema sent to the reviewer via ``ctx.elicit()``.

    MCP elicitation schemas must be flat with primitive fields, so every value
    is a string unless a numeric type is explicitly known.
    """
    definitions: dict[str, Any] = {}
    for field in fields:
        annotation = _FIELD_TYPES.get(field, str)
        definitions[field] = (
            annotation | None,
            Field(default=None, description=f"Value for the missing '{field}' field"),
        )
    return create_model("MissingClinicalFields", **definitions)


def _finding(
    rule_id: str,
    severity: Severity,
    message: str,
    *,
    document: str | None = None,
    field: str | None = None,
    weight: int = 0,
    blocking: bool = False,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "severity": severity.value,
        "message": message,
        "document": document,
        "field": field,
        "weight": weight,
        "blocking": blocking,
        "resolved": False,
    }


async def validate_completeness(
    ctx,
    packet: dict[str, Any],
    *,
    elicit: bool = True,
) -> dict[str, Any]:
    """Completeness-check a discharge packet, eliciting non-blocking gaps.

    ``packet`` maps doc types to extracted documents:
    ``{"discharge_report": {...}, "lab_report": {...}, "bill": {...}}``.
    """
    settings = get_settings()
    weights = settings.rules["risk_scoring_matrix"]["weights"]

    findings: list[dict[str, Any]] = []
    missing_nonblocking: list[str] = []
    total_required = 0
    total_present = 0

    for doc_type in (
        DocType.DISCHARGE_REPORT.value,
        DocType.LAB_REPORT.value,
        DocType.BILL.value,
    ):
        document = packet.get(doc_type)
        spec = DOCUMENT_REQUIREMENTS[doc_type]
        total_required += len(spec["required"])

        if document is None:
            findings.append(
                _finding(
                    "missing_document",
                    Severity.CRITICAL,
                    f"No {doc_type.replace('_', ' ')} was supplied for this case.",
                    document=doc_type,
                    weight=weights.get("missing_mandatory_field", 3),
                    blocking=True,
                )
            )
            continue

        missing, blocking = check_completeness(document, doc_type)
        total_present += len(spec["required"]) - len(missing)

        for field in missing:
            is_blocking = field in blocking
            weight_key = SOFT_FIELD_WEIGHTS.get(field, "missing_mandatory_field")
            findings.append(
                _finding(
                    f"missing_field.{doc_type}.{field}",
                    Severity.CRITICAL if is_blocking else Severity.WARNING,
                    f"{doc_type.replace('_', ' ').title()} is missing '{field}'."
                    + (" Discharge cannot be released." if is_blocking else ""),
                    document=doc_type,
                    field=field,
                    weight=weights.get(weight_key, 3),
                    blocking=is_blocking,
                )
            )
            if not is_blocking and field in ELICITABLE_FIELDS:
                missing_nonblocking.append(field)

    discharge = packet.get(DocType.DISCHARGE_REPORT.value) or {}
    prescription_issues = check_prescriptions(discharge.get("medications") or [])
    for issue in prescription_issues:
        findings.append(
            _finding(
                "incomplete_prescription_fields",
                Severity.CRITICAL if issue["blocking"] else Severity.WARNING,
                f"Prescription '{issue['medicine_name']}' is missing: "
                f"{', '.join(issue['missing_fields'])}.",
                document=DocType.DISCHARGE_REPORT.value,
                field=f"medications[{issue['row']}]",
                weight=weights.get("incomplete_prescription_fields", 4),
                blocking=bool(issue["blocking"]),
            )
        )

    elicitation: dict[str, Any] | None = None
    if elicit and missing_nonblocking:
        elicitation = await _elicit_missing_fields(ctx, sorted(set(missing_nonblocking)))
        if elicitation["action"] == ElicitAction.ACCEPT.value:
            _apply_elicited_values(packet, findings, elicitation["response"])

    completeness_score = round(100 * total_present / max(total_required, 1), 1)
    blocked = any(f["blocking"] and not f["resolved"] for f in findings)

    _log.info(
        "completeness validated",
        extra={
            "findings": len(findings),
            "completeness_score": completeness_score,
            "blocked": blocked,
            "elicited": elicitation["action"] if elicitation else None,
        },
    )
    return {
        "findings": findings,
        "completeness_score": completeness_score,
        "discharge_blocked": blocked,
        "elicitation": elicitation,
        "rules_version": settings.rules_version,
    }


async def _elicit_missing_fields(ctx, fields: list[str]) -> dict[str, Any]:
    """Ask the reviewer for non-blocking gaps; handle all three outcomes."""
    schema = build_elicitation_schema(fields)
    message = (
        "The discharge packet is missing "
        f"{len(fields)} non-blocking field(s): {', '.join(fields)}. "
        "Supply the values to complete this case now, or decline to send it to "
        "human review."
    )

    record: dict[str, Any] = {
        "fields_requested": fields,
        "requested_schema": schema.model_json_schema(),
        "action": ElicitAction.DECLINE.value,
        "response": {},
        "note": "",
    }

    try:
        result = await ctx.elicit(message=message, schema=schema)
    except Exception as exc:  # noqa: BLE001 - timeout or unsupported client
        _log.warning("elicitation unavailable; treating as decline", extra={"error": str(exc)})
        record["note"] = f"Elicitation unavailable ({exc}); treated as decline."
        return record

    action = getattr(result, "action", ElicitAction.DECLINE.value)

    if action == ElicitAction.ACCEPT.value:
        data = result.data.model_dump(exclude_none=True) if result.data else {}
        record["action"] = ElicitAction.ACCEPT.value
        record["response"] = data
        record["note"] = f"Reviewer supplied {len(data)} of {len(fields)} field(s)."
    elif action == ElicitAction.CANCEL.value:
        record["action"] = ElicitAction.CANCEL.value
        record["note"] = "Reviewer cancelled; case aborted and escalated to HITL."
    else:
        record["action"] = ElicitAction.DECLINE.value
        record["note"] = "Reviewer declined; gaps left unresolved and flagged for HITL."

    _log.info("elicitation resolved", extra={"action": record["action"], "fields": fields})
    return record


def _apply_elicited_values(
    packet: dict[str, Any], findings: list[dict[str, Any]], values: dict[str, Any]
) -> None:
    """Write accepted reviewer input back into the packet and close its findings."""
    for field, value in values.items():
        if value in (None, ""):
            continue
        for doc_type, document in packet.items():
            if not isinstance(document, dict):
                continue
            spec = DOCUMENT_REQUIREMENTS.get(doc_type)
            if spec and field in spec["required"] and _is_missing(document.get(field)):
                document[field] = value
                for finding in findings:
                    if finding["rule_id"] == f"missing_field.{doc_type}.{field}":
                        finding["resolved"] = True
                        finding["resolution_note"] = "Supplied by reviewer via MCP elicitation."


def register(mcp) -> None:
    @mcp.tool(
        name="clinical_rules_engine",
        description=(
            "Validate a discharge packet for completeness against Table 3 of "
            "the clinical rules. Non-blocking gaps trigger an MCP elicitation "
            "request to the reviewer; blocking gaps escalate to HITL."
        ),
    )
    async def clinical_rules_engine(
        ctx: Context, packet: dict[str, Any], elicit: bool = True
    ) -> dict[str, Any]:
        return await validate_completeness(ctx, packet, elicit=elicit)

    @mcp.tool(
        name="get_document_requirements",
        description="Return the required and blocking field matrix for each "
        "document type (doc Table 3).",
    )
    def get_document_requirements() -> dict[str, Any]:
        return {
            doc_type: {"required": list(spec["required"]), "blocking": list(spec["blocking"])}
            for doc_type, spec in DOCUMENT_REQUIREMENTS.items()
        }
