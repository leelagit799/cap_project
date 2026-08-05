"""Unit tests for dynamic patient document uploads."""

from __future__ import annotations

from pathlib import Path

import pytest

from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.ingest.registry import PatientUploadRegistry
from hospital_ai.ingest.service import UploadService
from hospital_ai.ingest.storage import infer_doc_type, target_filename


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    upload_dir = tmp_path / "input"
    state_dir = tmp_path / "state"
    upload_dir.mkdir()
    state_dir.mkdir()
    monkeypatch.setenv("AGENT_AUTH_TOKEN", "test-token-for-uploads")
    from hospital_ai.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    object.__setattr__(settings, "upload_dir", upload_dir)
    object.__setattr__(settings, "state_dir", state_dir)
  # registry uses state_dir for sqlite
    return upload_dir, state_dir


class TestNaming:
    def test_doctor_report_uses_physician_name(self):
        assert target_filename("P1025", "discharge_report", "blood.pdf", doctor_name="Dr Smith") == "P1025_DrSmith.pdf"

    def test_doctor_report_defaults_to_doctor(self):
        assert target_filename("P1025", "discharge_report", "note.txt") == "P1025_doctor.txt"

    def test_lab_and_bill_patterns(self):
        assert target_filename("P1025", "lab_report", "scan.png") == "P1025_labs.png"
        assert target_filename("P1025", "bill", "invoice.json") == "P1025_bill.json"

    def test_infer_doc_type_in_patient_folder(self, upload_env):
        upload_dir, _ = upload_env
        folder = upload_dir / "P1099"
        folder.mkdir()
        assert infer_doc_type(folder / "P1099_labs.pdf") == "lab_report"
        assert infer_doc_type(folder / "P1099_bill.json") == "bill"
        assert infer_doc_type(folder / "P1099_DrLee.docx") == "discharge_report"


class TestRegistry:
    def test_ids_continue_after_workspace_max(self, upload_env, tmp_path):
        upload_dir, state_dir = upload_env
        (upload_dir / "P1024").mkdir()
        (upload_dir / "P1024" / "P1024_labs.txt").write_text("lab", encoding="utf-8")

        registry = PatientUploadRegistry(state_dir / "registry.sqlite")
        first = registry.allocate_patient_id()
        second = registry.allocate_patient_id()
        assert first["patient_id"] == "P1025"
        assert second["patient_id"] == "P1026"

    def test_ids_persist_across_instances(self, upload_env):
        _, state_dir = upload_env
        db = state_dir / "registry.sqlite"
        first = PatientUploadRegistry(db).allocate_patient_id()["patient_id"]
        second = PatientUploadRegistry(db).allocate_patient_id()["patient_id"]
        assert first == "P1025"
        assert second == "P1026"


class TestUploadService:
    def test_batch_save_all_documents(self, upload_env):
        upload_dir, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "batch.sqlite"))
        patient_id = service.create_patient(doctor_name="Dr Patel")["patient_id"]

        result = service.save_patient_documents(
            {
                "discharge_report": ("notes.pdf", b"%PDF-1.4"),
                "lab_report": ("labs.png", b"\x89PNG\r\n"),
                "bill": ("bill.json", b'{"total": 1}'),
            },
            patient_id=patient_id,
            doctor_name="Dr Patel",
        )

        assert result["patient_id"] == patient_id
        assert result["complete"] is True
        folder = upload_dir / patient_id
        assert (folder / f"{patient_id}_DrPatel.pdf").is_file()
        assert (folder / f"{patient_id}_labs.png").is_file()
        assert (folder / f"{patient_id}_bill.json").is_file()

    def test_batch_save_allocates_patient_id_when_missing(self, upload_env):
        upload_dir, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "alloc.sqlite"))

        result = service.save_patient_documents(
            {
                "discharge_report": ("notes.pdf", b"%PDF-1.4"),
                "lab_report": ("labs.png", b"\x89PNG\r\n"),
                "bill": ("bill.json", b'{"total": 1}'),
            },
            doctor_name="Dr Lee",
        )

        patient_id = result["patient_id"]
        assert patient_id.startswith("P")
        assert (upload_dir / patient_id).is_dir()

    def test_batch_save_rejects_incomplete_packet(self, upload_env):
        _, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "incomplete.sqlite"))
        patient_id = service.create_patient()["patient_id"]
        with pytest.raises(DischargeFlowError):
            service.save_patient_documents(
                {
                    "discharge_report": ("notes.pdf", b"%PDF"),
                    "lab_report": ("labs.png", b"\x89PNG"),
                },
                patient_id=patient_id,
            )

    def test_batch_save_rolls_back_on_validation_error(self, upload_env):
        upload_dir, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "rollback.sqlite"))
        patient_id = service.create_patient()["patient_id"]

        with pytest.raises(DischargeFlowError):
            service.save_patient_documents(
                {
                    "discharge_report": ("notes.pdf", b"%PDF"),
                    "lab_report": ("labs.png", b"\x89PNG"),
                    "bill": ("bill.docx", b"bad"),
                },
                patient_id=patient_id,
            )

        folder = upload_dir / patient_id
        saved = list(folder.glob(f"{patient_id}_*")) if folder.is_dir() else []
        assert not saved

    def test_upload_rename_and_process_gate(self, upload_env):
        upload_dir, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "svc.sqlite"))
        patient_id = service.create_patient(doctor_name="Dr Patel")["patient_id"]

        service.upload_document(
            patient_id, "discharge_report", "notes.pdf", b"%PDF-1.4", doctor_name="Dr Patel"
        )
        service.upload_document(patient_id, "lab_report", "labs.png", b"\x89PNG\r\n")
        service.upload_document(patient_id, "bill", "bill.json", b'{"total": 1}')

        folder = upload_dir / patient_id
        assert (folder / f"{patient_id}_DrPatel.pdf").is_file()
        assert (folder / f"{patient_id}_labs.png").is_file()
        assert (folder / f"{patient_id}_bill.json").is_file()

        detail = service.get_patient(patient_id)
        assert detail["complete"] is True

    def test_duplicate_doc_type_rejected(self, upload_env):
        _, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "dup.sqlite"))
        patient_id = service.create_patient()["patient_id"]
        service.upload_document(patient_id, "lab_report", "a.pdf", b"%PDF")
        with pytest.raises(DischargeFlowError):
            service.upload_document(patient_id, "lab_report", "b.pdf", b"%PDF")

    def test_replace_allows_new_lab_file(self, upload_env):
        _, state_dir = upload_env
        service = UploadService(PatientUploadRegistry(state_dir / "rep.sqlite"))
        patient_id = service.create_patient()["patient_id"]
        service.upload_document(patient_id, "lab_report", "a.pdf", b"%PDF-1")
        service.upload_document(patient_id, "lab_report", "b.pdf", b"%PDF-2", replace=True)
        docs = service.get_patient(patient_id)["documents"]
        assert len(docs) == 1
        assert docs[0]["filename"].endswith("_labs.pdf")
