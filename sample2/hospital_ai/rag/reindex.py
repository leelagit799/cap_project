"""Re-index corrected clinical records into the RAG vector store."""

from __future__ import annotations

from typing import Any

from hospital_ai.rag.roles import IndexingAgent
from hospital_ai.rag.store import FaissStore, get_store
from hospital_ai.storage import CaseStore


def resolve_record_patient_id(
    record: dict[str, Any],
    *,
    case_id: str | None = None,
    store: CaseStore | None = None,
) -> str:
    """Return the patient id for chunk filtering, even on partial records."""
    patient_id = record.get("patient_id")
    if patient_id:
        return str(patient_id)

    discharge = record.get("discharge_report") or {}
    if discharge.get("patient_id"):
        return str(discharge["patient_id"])

    if case_id and store is not None:
        case = store.get_case(case_id)
        if case and case.get("patient_id"):
            return str(case["patient_id"])

    return "unknown"


def reindex_case_record(
    case_id: str,
    record: dict[str, Any],
    *,
    store: FaissStore | None = None,
    case_store: CaseStore | None = None,
) -> dict[str, Any]:
    """Index (or replace) all RAG chunks for one corrected case record."""
    payload = dict(record)
    payload["patient_id"] = resolve_record_patient_id(
        payload, case_id=case_id, store=case_store
    )
    return IndexingAgent(store=store or get_store()).index_case(case_id, payload)
