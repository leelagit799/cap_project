"""Helpers for building HITL correction payloads from dashboard form state."""

from __future__ import annotations

import math
import re
from typing import Any

_MEDICATION_RULE_IDS = {
    "med_omission_check",
    "incomplete_prescription_fields",
    "allergy_contradiction_check",
    "high_risk_med_missing_in_ehr",
}
_PRESCRIPTION_ROW_FIELD = re.compile(r"medications\[(\d+)\]")

#: Fields the rules engine may elicit; values land on the discharge report.
_ELICITABLE_DISCHARGE_FIELDS = {
    "address",
    "gender",
    "age",
    "ward",
    "bed_no",
    "admission_date",
    "discharge_date",
    "doctors",
    "follow_up_appointments",
    "discharge_instructions",
}


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def normalize_medication_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure prescription rows keep medicine_name and JSON-safe values."""
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        med = {key: _clean_value(value) for key, value in dict(row).items()}
        medicine_name = med.get("medicine_name") or med.get("name")
        if not medicine_name:
            continue
        med["medicine_name"] = str(medicine_name)
        med.pop("name", None)
        if med.get("sl_no") is not None:
            try:
                med["sl_no"] = int(med["sl_no"])
            except (TypeError, ValueError):
                pass
        if med.get("total_quantity") is not None:
            med["total_quantity"] = str(med["total_quantity"])
        cleaned.append(med)
    return cleaned


def corrections_from_elicitation(answers: dict[str, Any]) -> dict[str, Any]:
    """Map MCP elicitation answers onto dotted record correction paths."""
    corrections: dict[str, Any] = {}
    for field, value in answers.items():
        cleaned = _clean_value(value)
        if cleaned is None:
            continue
        if field in _ELICITABLE_DISCHARGE_FIELDS:
            if field == "follow_up_appointments":
                corrections["discharge_report.follow_up_appointments"] = (
                    [cleaned] if isinstance(cleaned, str) else list(cleaned)
                )
            elif field == "age":
                corrections["discharge_report.age"] = (
                    int(cleaned) if str(cleaned).isdigit() else cleaned
                )
            elif field == "doctors":
                physician = cleaned[0] if isinstance(cleaned, list) else str(cleaned)
                corrections["discharge_report.attending_physician"] = physician
                corrections["discharge_report.discharge_approved_by"] = physician
                corrections["discharge_report.discharge_approved"] = True
            else:
                corrections[f"discharge_report.{field}"] = cleaned
    return corrections


def merge_corrections(*parts: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged


def medication_corrections_if_changed(
    stored: list[dict[str, Any]] | None,
    edited: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return a correction payload when the editor differs from the saved record."""
    stored_norm = normalize_medication_rows(stored or [])
    current_norm = merge_medication_name_edits(stored_norm, edited or [])
    if current_norm != stored_norm:
        return {"discharge_report.medications": current_norm}
    return {}


def merge_medication_name_edits(
    stored: list[dict[str, Any]] | None,
    edited: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Apply medicine-name edits from the HITL table onto the stored prescription rows."""
    stored_norm = normalize_medication_rows(stored or [])
    edited_norm = normalize_medication_rows(edited or [])
    merged: list[dict[str, Any]] = []
    for index, edited_row in enumerate(edited_norm):
        base = dict(stored_norm[index]) if index < len(stored_norm) else {}
        if edited_row.get("medicine_name"):
            base["medicine_name"] = edited_row["medicine_name"]
        for key, value in edited_row.items():
            if key == "medicine_name" or value in (None, ""):
                continue
            base[key] = value
        if base.get("medicine_name"):
            merged.append(base)
    return normalize_medication_rows(merged)


def _canonical_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _is_plausible_drug_name(name: str) -> bool:
    cleaned = str(name or "").strip()
    if not cleaned or len(cleaned) > 80:
        return False
    lowered = cleaned.lower()
    return " is a " not in lowered and " medication " not in lowered


def _display_drug_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not _is_plausible_drug_name(cleaned):
        match = re.search(r"'([^']+)'", cleaned)
        if match and _is_plausible_drug_name(match.group(1)):
            cleaned = match.group(1)
        else:
            cleaned = cleaned.split()[0] if cleaned.split() else cleaned
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _parse_missing_prescription_fields(message: str) -> list[str]:
    marker = " is missing: "
    if marker not in message:
        return []
    tail = message.split(marker, 1)[1].rstrip(".")
    return [field.strip() for field in tail.split(",") if field.strip()]


def medication_correction_suggestions(
    findings: list[dict[str, Any]],
    medications: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build reviewer-facing medication correction hints from validation findings."""
    medications = medications or []
    existing = {_canonical_name(m.get("medicine_name", "")) for m in medications}
    suggestions: list[dict[str, Any]] = []
    seen: set[str] = set()

    for finding in findings:
        if finding.get("resolved"):
            continue
        rule_id = str(finding.get("rule_id", ""))
        if rule_id not in _MEDICATION_RULE_IDS:
            continue

        suggestion_id = f"{rule_id}:{finding.get('field')}:{finding.get('message')}"
        if suggestion_id in seen:
            continue
        seen.add(suggestion_id)

        if rule_id == "med_omission_check":
            expected = finding.get("expected")
            actual = finding.get("actual")
            if expected and _is_plausible_drug_name(str(expected)):
                canonical = _canonical_name(str(expected))
                if canonical in existing:
                    continue
                display_name = _display_drug_name(str(expected))
                suggestions.append(
                    {
                        "id": suggestion_id,
                        "title": f"Add '{display_name}' to the prescription table",
                        "detail": finding["message"],
                        "action": {
                            "type": "add_row",
                            "row": {"medicine_name": display_name},
                        },
                    }
                )
            elif actual and _is_plausible_drug_name(str(actual)):
                suggestions.append(
                    {
                        "id": suggestion_id,
                        "title": f"Verify '{actual}' against the EHR medication list",
                        "detail": finding["message"],
                        "action": None,
                    }
                )
            continue

        if rule_id == "incomplete_prescription_fields":
            missing_fields = _parse_missing_prescription_fields(str(finding.get("message", "")))
            row_index = None
            field_ref = finding.get("field")
            if field_ref:
                match = _PRESCRIPTION_ROW_FIELD.search(str(field_ref))
                if match:
                    row_index = int(match.group(1)) - 1
            medicine_name = None
            if row_index is not None and 0 <= row_index < len(medications):
                medicine_name = medications[row_index].get("medicine_name")
            if medicine_name is None:
                prefix = "Prescription '"
                message = str(finding.get("message", ""))
                if prefix in message:
                    medicine_name = message.split(prefix, 1)[1].split("'", 1)[0]
            title = (
                f"Complete missing fields for '{medicine_name}'"
                if medicine_name
                else "Complete missing prescription fields"
            )
            suggestions.append(
                {
                    "id": suggestion_id,
                    "title": title,
                    "detail": finding["message"],
                    "action": {
                        "type": "fill_row",
                        "row_index": row_index,
                        "medicine_name": medicine_name,
                        "fields": missing_fields,
                    }
                    if missing_fields
                    else None,
                }
            )
            continue

        if rule_id in {"allergy_contradiction_check", "high_risk_med_missing_in_ehr"}:
            actual = finding.get("actual")
            if actual:
                suggestions.append(
                    {
                        "id": suggestion_id,
                        "title": (
                            f"Replace or remove '{actual}'"
                            if rule_id == "allergy_contradiction_check"
                            else f"Confirm high-risk medication '{actual}' with pharmacy/EHR"
                        ),
                        "detail": finding["message"],
                        "action": {
                            "type": "remove_row",
                            "medicine_name": str(actual),
                        }
                        if rule_id == "allergy_contradiction_check"
                        else None,
                    }
                )

    return suggestions


def apply_medication_suggestion(
    medications: list[dict[str, Any]],
    action: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply one medication suggestion onto the current prescription rows."""
    action_type = action.get("type")
    rows = [dict(row) for row in medications]

    if action_type == "add_row":
        row = dict(action.get("row") or {})
        if row.get("medicine_name"):
            rows.append(row)
        return normalize_medication_rows(rows)

    if action_type == "remove_row":
        target = _canonical_name(str(action.get("medicine_name", "")))
        filtered = [
            row
            for row in rows
            if _canonical_name(str(row.get("medicine_name", ""))) != target
        ]
        return normalize_medication_rows(filtered)

    if action_type == "fill_row":
        missing_fields = list(action.get("fields") or [])
        if not missing_fields:
            return normalize_medication_rows(rows)

        row_index = action.get("row_index")
        target_name = action.get("medicine_name")
        updated = False
        for index, row in enumerate(rows):
            if row_index is not None and index != row_index:
                continue
            if target_name and row.get("medicine_name") != target_name:
                continue
            for field in missing_fields:
                if _clean_value(row.get(field)) is None:
                    row[field] = _default_prescription_value(field)
            updated = True
            if row_index is not None:
                break
        if not updated and target_name:
            for row in rows:
                if row.get("medicine_name") == target_name:
                    for field in missing_fields:
                        if _clean_value(row.get(field)) is None:
                            row[field] = _default_prescription_value(field)
        return normalize_medication_rows(rows)

    return normalize_medication_rows(rows)


def _default_prescription_value(field: str) -> Any:
    defaults = {
        "sl_no": 1,
        "strength": "—",
        "dosage": "As directed",
        "frequency": "OD",
        "route": "ORAL",
        "period": "30 days",
        "remarks": "Filled during HITL review",
        "total_quantity": "—",
    }
    return defaults.get(field, "—")
