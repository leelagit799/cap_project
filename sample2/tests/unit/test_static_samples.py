"""Unit tests for static sample patient document management."""

from __future__ import annotations

from pathlib import Path

import pytest

from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.core.schemas import ValidationResult
from hospital_ai.ingest.static_samples import (
    SAMPLE_PATIENT_IDS,
    get_patient,
    list_sample_patients,
    update_patient_documents,
)


@pytest.fixture
def static_env(tmp_path, monkeypatch):
    incoming = tmp_path / "incoming"
    for folder in ("doctor_reports", "lab_reports", "bills"):
        (incoming / folder).mkdir(parents=True)

    (incoming / "doctor_reports" / "P1019_thomas_wright.txt").write_text("doctor", encoding="utf-8")
    (incoming / "lab_reports" / "P1019_labs.txt").write_text("labs", encoding="utf-8")
    (incoming / "bills" / "P1019_bill.json").write_text('{"total": 1}', encoding="utf-8")

    monkeypatch.setenv("AGENT_AUTH_TOKEN", "test-token-for-static-samples")
    from hospital_ai.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    object.__setattr__(settings.roots, "workspace", incoming)
    return incoming


class TestStaticSamples:
    def test_list_sample_patients(self):
        assert list_sample_patients() == list(SAMPLE_PATIENT_IDS)

    def test_get_patient_complete(self, static_env):
        detail = get_patient("P1019")
        assert detail is not None
        assert detail["complete"] is True
        assert {doc["doc_type"] for doc in detail["documents"]} == {
            "discharge_report",
            "lab_report",
            "bill",
        }

    def test_replace_single_document(self, static_env):
        update_patient_documents(
            "P1019",
            replacements={"lab_report": ("labs.pdf", b"%PDF-1.4 lab")},
        )
        detail = get_patient("P1019")
        assert detail is not None
        lab = next(doc for doc in detail["documents"] if doc["doc_type"] == "lab_report")
        assert lab["filename"] == "P1019_labs.pdf"
        assert not (static_env / "lab_reports" / "P1019_labs.txt").exists()

    def test_replace_doctor_report_uses_physician_name(self, static_env):
        update_patient_documents(
            "P1019",
            replacements={"discharge_report": ("note.pdf", b"%PDF-1.4")},
            doctor_name="Dr Smith",
        )
        detail = get_patient("P1019")
        assert detail is not None
        doctor = next(doc for doc in detail["documents"] if doc["doc_type"] == "discharge_report")
        assert doctor["filename"] == "P1019_DrSmith.pdf"

    def test_delete_without_replacement_rejected(self, static_env):
        with pytest.raises(DischargeFlowError, match="Deleted documents must be replaced"):
            update_patient_documents(
                "P1019",
                replacements={},
                deleted_types={"lab_report"},
            )

    def test_incomplete_packet_rejected(self, static_env):
        (static_env / "lab_reports" / "P1019_labs.txt").unlink()
        (static_env / "bills" / "P1019_bill.json").unlink()
        with pytest.raises(DischargeFlowError, match="All three documents are required"):
            update_patient_documents(
                "P1019",
                replacements={"lab_report": ("labs.pdf", b"%PDF-1.4")},
            )


class TestCaseStoreReset:
    def test_delete_patient_cases(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_AUTH_TOKEN", "test-token-for-case-reset")
        from hospital_ai.core.config import get_settings
        from hospital_ai.storage import CaseStore

        get_settings.cache_clear()
        settings = get_settings()
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        object.__setattr__(settings, "state_dir", state_dir)

        store = CaseStore(state_dir / "cases.sqlite")
        store.create_case("CASE-P1019-A", "P1019", "trace-a")
        store.create_case("CASE-P1020-A", "P1020", "trace-b")
        store.save_validation(
            "CASE-P1019-A",
            ValidationResult(case_id="CASE-P1019-A", patient_id="P1019"),
        )

        deleted = store.delete_patient_cases("P1019")
        assert deleted == ["CASE-P1019-A"]
        assert store.list_cases(patient_id="P1019") == []
        assert len(store.list_cases(patient_id="P1020")) == 1
