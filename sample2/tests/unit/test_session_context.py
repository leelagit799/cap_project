"""Session context helpers for the Streamlit dashboard."""

from __future__ import annotations

from hospital_ai.ui.streamlit_hitl.session_context import (
    activate_from_outcome,
    clear_active_context,
    get_active_context,
    refresh_active_from_case,
    set_pending_process,
)


class TestSessionContext:
    def test_activate_and_clear(self):
        clear_active_context()
        activate_from_outcome(
            {
                "patient_id": "P1025",
                "case_id": "CASE-P1025-1",
                "trace_id": "abc123",
                "status": "HITL_PENDING",
                "risk_level": "High",
                "requires_hitl": True,
            },
            patient_name="Jane Doe",
        )
        ctx = get_active_context()
        assert ctx is not None
        assert ctx["patient_id"] == "P1025"
        assert ctx["patient_name"] == "Jane Doe"
        assert ctx["case_id"] == "CASE-P1025-1"
        clear_active_context()
        assert get_active_context() is None

    def test_pending_process_flag(self):
        set_pending_process("P1099")
        from hospital_ai.ui.streamlit_hitl.session_context import pending_process_patient

        assert pending_process_patient() == "P1099"
        set_pending_process(None)
        assert pending_process_patient() is None
