"""Shared foundation: configuration, contracts, logging, retries, identifiers."""

from hospital_ai.core.config import Settings, get_settings
from hospital_ai.core.ids import extract_patient_id, new_case_id, new_trace_id, utc_now_iso
from hospital_ai.core.logging import configure_logging, get_logger

__all__ = [
    "Settings",
    "configure_logging",
    "extract_patient_id",
    "get_logger",
    "get_settings",
    "new_case_id",
    "new_trace_id",
    "utc_now_iso",
]
