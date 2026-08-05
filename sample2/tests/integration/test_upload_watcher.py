"""Integration tests — uploaded documents discovered by the clinical watcher."""

from __future__ import annotations

import mcp.types as types
import pytest
from pydantic import AnyUrl

from hospital_ai.ingest.registry import PatientUploadRegistry
from hospital_ai.ingest.service import UploadService
from hospital_ai.ingest.storage import infer_doc_type
from hospital_ai.mcp_servers.primary.tools.watcher import scan_clinical_workspace

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    upload_dir = tmp_path / "input"
    static_dir = tmp_path / "incoming"
    state_dir = tmp_path / "state"
    for path in (upload_dir, static_dir, state_dir):
        path.mkdir(parents=True)
    monkeypatch.setenv("AGENT_AUTH_TOKEN", "test-token-for-uploads")
    from hospital_ai.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    object.__setattr__(settings, "upload_dir", upload_dir)
    object.__setattr__(settings, "state_dir", state_dir)
    object.__setattr__(settings.roots, "workspace", static_dir)
    return upload_dir, static_dir, state_dir


class TestWatcherUploadIntegration:
    async def test_uploaded_patient_folder_is_discovered(self, upload_env):
        upload_dir, _, state_dir = upload_env
        registry = PatientUploadRegistry(state_dir / "watch.sqlite")
        service = UploadService(registry)

        patient_id = service.create_patient(doctor_name="Dr Kim")["patient_id"]
        service.upload_document(
            patient_id, "discharge_report", "d.pdf", b"%PDF", doctor_name="Dr Kim"
        )
        service.upload_document(patient_id, "lab_report", "l.txt", b"lab")
        service.upload_document(patient_id, "bill", "b.json", b"{}")

        class FakeCtx:
            class session:
                @staticmethod
                async def list_roots():
                    return types.ListRootsResult(
                        roots=[
                            types.Root(
                                uri=AnyUrl(upload_dir.resolve().as_uri()),
                                name="uploads",
                            )
                        ]
                    )

        payload = await scan_clinical_workspace(FakeCtx(), patient_id=patient_id)

        assert payload["patient_count"] == 1
        entry = payload["patients"][0]
        assert entry["patient_id"] == patient_id
        assert entry["complete"] is True
        assert set(entry["doc_types"]) == {"discharge_report", "lab_report", "bill"}

    def test_infer_matches_watcher_expectations(self, upload_env):
        upload_dir, _, _ = upload_env
        folder = upload_dir / "P1100"
        folder.mkdir()
        for name, expected in (
            ("P1100_DrKim.pdf", "discharge_report"),
            ("P1100_labs.txt", "lab_report"),
            ("P1100_bill.json", "bill"),
        ):
            path = folder / name
            path.write_text("x", encoding="utf-8")
            assert infer_doc_type(path) == expected
