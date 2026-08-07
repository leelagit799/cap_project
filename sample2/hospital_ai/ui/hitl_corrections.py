"""Helpers for building HITL correction payloads from dashboard form state."""

from __future__ import annotations

import math
from typing import Any

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
            else:
                corrections[f"discharge_report.{field}"] = cleaned
    return corrections


def merge_corrections(*parts: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged
