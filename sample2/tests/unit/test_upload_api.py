"""API tests for the patient upload service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hospital_ai.ingest.app import create_app
from hospital_ai.ingest.registry import PatientUploadRegistry


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    upload_dir = tmp_path / "input"
    state_dir = tmp_path / "state"
    upload_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    monkeypatch.setenv("AGENT_AUTH_TOKEN", "test-token-for-uploads")
    from hospital_ai.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    object.__setattr__(settings, "upload_dir", upload_dir)
    object.__setattr__(settings, "state_dir", state_dir)
    return upload_dir, state_dir


@pytest.fixture
def client(upload_env):
    return TestClient(create_app())


class TestUploadAPI:
    def test_create_list_upload_delete(self, client, upload_env):
        created = client.post("/patients", params={"doctor_name": "Dr Lee"}).json()
        patient_id = created["patient_id"]
        assert patient_id.startswith("P")

        files = {"file": ("labs.pdf", b"%PDF-1.4", "application/pdf")}
        uploaded = client.post(
            f"/patients/{patient_id}/upload",
            params={"doc_type": "lab_report"},
            files=files,
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["filename"] == f"{patient_id}_labs.pdf"

        detail = client.get(f"/patients/{patient_id}").json()
        assert len(detail["documents"]) == 1

        deleted = client.delete(f"/patients/{patient_id}/documents/{patient_id}_labs.pdf")
        assert deleted.status_code == 200
        assert client.get(f"/patients/{patient_id}").json()["documents"] == []

    def test_rejects_invalid_extension(self, client):
        patient_id = client.post("/patients").json()["patient_id"]
        response = client.post(
            f"/patients/{patient_id}/upload",
            params={"doc_type": "lab_report"},
            files={"file": ("bad.docx", b"x", "application/octet-stream")},
        )
        assert response.status_code == 400
