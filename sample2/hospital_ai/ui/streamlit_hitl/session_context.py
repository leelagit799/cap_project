"""Active processing context for the HITL dashboard.

After **Process Patient** runs on the Document Viewer, the resulting case is
stored in Streamlit session state. Validation Report and HITL Corrections always
operate on this context — reviewers cannot switch patients mid-review.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from hospital_ai.ui.streamlit_hitl import icons

_CONTEXT_KEY = "active_processing_context"
_CONFIRM_KEY = "awaiting_process_confirm"


def get_active_context() -> dict[str, Any] | None:
    return st.session_state.get(_CONTEXT_KEY)


def set_active_context(context: dict[str, Any]) -> None:
    st.session_state[_CONTEXT_KEY] = context


def clear_active_context() -> None:
    st.session_state.pop(_CONTEXT_KEY, None)


def pending_process_patient() -> str | None:
    return st.session_state.get(_CONFIRM_KEY)


def set_pending_process(patient_id: str | None) -> None:
    if patient_id is None:
        st.session_state.pop(_CONFIRM_KEY, None)
    else:
        st.session_state[_CONFIRM_KEY] = patient_id


def activate_from_outcome(
    outcome: dict[str, Any],
    *,
    patient_name: str | None = None,
) -> None:
    """Record the case that just finished processing."""
    set_active_context(
        {
            "patient_id": outcome["patient_id"],
            "patient_name": patient_name or outcome["patient_id"],
            "case_id": outcome["case_id"],
            "trace_id": outcome.get("trace_id") or "",
            "workflow_status": outcome.get("status") or "UNKNOWN",
            "current_agent": "host-orchestrator",
            "risk_level": outcome.get("risk_level"),
            "requires_hitl": outcome.get("requires_hitl", False),
        }
    )
    set_pending_process(None)


def refresh_active_from_case(case: dict[str, Any], *, patient_name: str | None = None) -> None:
    """Sync workflow fields after a re-validation or status change."""
    ctx = get_active_context() or {}
    set_active_context(
        {
            "patient_id": case["patient_id"],
            "patient_name": patient_name or ctx.get("patient_name") or case["patient_id"],
            "case_id": case["case_id"],
            "trace_id": case.get("trace_id") or ctx.get("trace_id") or "",
            "workflow_status": case.get("status") or ctx.get("workflow_status") or "UNKNOWN",
            "current_agent": ctx.get("current_agent") or "clinical-validator",
            "risk_level": case.get("risk_level"),
            "requires_hitl": case.get("status") == "HITL_PENDING",
        }
    )


def render_active_panel() -> bool:
    """Read-only banner for the patient currently in workflow. Returns False if none."""
    ctx = get_active_context()
    if not ctx:
        return False

    name = ctx.get("patient_name") or "—"
    st.markdown(
        '<div class="df-card"><h3>Active processing context</h3>'
        f'<span class="df-pill">{icons.svg("user", size=14)} {ctx.get("patient_id")}</span>'
        f'<span class="df-pill">{name}</span>'
        f'<span class="df-pill">Case {ctx.get("case_id")}</span>'
        f'<span class="df-pill">Trace {str(ctx.get("trace_id") or "")[:16]}…</span>'
        f'<span class="df-pill">Status {ctx.get("workflow_status")}</span>'
        f'<span class="df-pill">Agent {ctx.get("current_agent") or "—"}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    return True


def require_active_case(svc) -> dict[str, Any] | None:
    """Return the active case row or show the standard empty-state message."""
    ctx = get_active_context()
    if not ctx:
        st.info(
            "No active patient is currently being processed. Please select a patient "
            "from the **Document Viewer** and click **Process Patient**."
        )
        return None

    case = svc.case(ctx["case_id"])
    if case is None:
        st.warning("The active case could not be found. Start processing again from Document Viewer.")
        clear_active_context()
        return None

    return case
