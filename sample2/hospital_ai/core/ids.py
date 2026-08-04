"""Identifier generation.

The Host Orchestrator mints one ``case_id`` and one ``trace_id`` per discharge
case (doc §7.2: "End-to-end trace ID per discharge case, passed through all
agents via message metadata").
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

# A trailing \b would fail on "P1019_labs.txt", since "_" is a word character.
PATIENT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])P\d{4}(?!\d)")


def new_trace_id() -> str:
    """A LangFuse-compatible trace id, unique per case."""
    return uuid.uuid4().hex


def new_case_id(patient_id: str) -> str:
    """Human-readable, sortable case id: ``CASE-P1019-20260804T063000-a1b2c3``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"CASE-{patient_id}-{stamp}-{uuid.uuid4().hex[:6]}"


def extract_patient_id(name: str) -> str | None:
    """Pull the ``P####`` patient id out of a filename or path.

    Returns None for files that carry no patient prefix; the Monitor Agent
    quarantines those rather than guessing.
    """
    match = PATIENT_ID_PATTERN.search(name)
    return match.group(0) if match else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
