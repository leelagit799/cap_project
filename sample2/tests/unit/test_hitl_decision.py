"""Unit tests for HITL discharge decision handling."""

from __future__ import annotations

import pytest

from hospital_ai.core.schemas import CaseStatus
from hospital_ai.storage import CaseStore
from hospital_ai.ui.service import DashboardService


@pytest.fixture
def dashboard_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_TOKEN", "test-token-hitl-decision")
    from hospital_ai.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    object.__setattr__(settings, "state_dir", state_dir)
    store = CaseStore(state_dir / "cases.sqlite")
    store.create_case("CASE-P1019", "P1019", "trace-1019")
    store.save_record(
        "CASE-P1019",
        {
            "discharge_report": {"patient_id": "P1019", "medications": []},
            "bill": {"payment_status": "UNPAID"},
        },
    )
    store.update_status("CASE-P1019", CaseStatus.HITL_PENDING)
    return store


class TestHitlDecision:
    def test_allow_overrides_blocked_discharge(self, dashboard_env, monkeypatch):
        store = dashboard_env
        store.set_discharge_gate(
            "CASE-P1019", discharge_blocked=True, status=CaseStatus.HITL_PENDING
        )

        svc = DashboardService(store=store)
        monkeypatch.setattr(svc, "reindex_case", lambda case_id: {"indexed": 0})
        monkeypatch.setattr(
            svc,
            "generate_summary",
            lambda case_id: {"ok": True, "summary": {"sections": []}},
        )

        outcome = svc.apply_hitl_decision(
            "CASE-P1019",
            "allow",
            corrections={"bill.payment_status": "PAID"},
            rerun_validation=True,
        )

        case = store.get_case("CASE-P1019")
        assert outcome["discharge_blocked"] is False
        assert outcome["status"] == CaseStatus.SUMMARY_READY.value
        assert case["discharge_blocked"] == 0
        assert case["status"] == CaseStatus.SUMMARY_READY.value
        assert case["risk_level"] == "Low"
        assert outcome["summary_generated"] is True
        assert store.get_record("CASE-P1019")["bill"]["payment_status"] == "PAID"

    def test_reject_blocks_discharge(self, dashboard_env, monkeypatch):
        store = dashboard_env
        store.update_status("CASE-P1019", CaseStatus.SUMMARY_READY)

        svc = DashboardService(store=store)
        monkeypatch.setattr(svc, "reindex_case", lambda case_id: {"indexed": 0})

        outcome = svc.apply_hitl_decision(
            "CASE-P1019",
            "reject",
            rerun_validation=True,
        )

        case = store.get_case("CASE-P1019")
        assert outcome["discharge_blocked"] is True
        assert case["status"] == CaseStatus.HITL_PENDING.value
