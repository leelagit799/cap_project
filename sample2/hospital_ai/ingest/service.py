"""Upload orchestration — storage, registry, and workflow trigger."""

from __future__ import annotations

from typing import Any

from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.core.logging import get_logger
from hospital_ai.ingest import storage
from hospital_ai.ingest.models import DocType
from hospital_ai.ingest.registry import PatientUploadRegistry

_log = get_logger(__name__, component="upload-service")

_REQUIRED_TYPES = {"discharge_report", "lab_report", "bill"}


class UploadService:
    def __init__(self, registry: PatientUploadRegistry | None = None) -> None:
        self.registry = registry or PatientUploadRegistry()

    def create_patient(self, *, doctor_name: str | None = None) -> dict[str, Any]:
        return self.registry.allocate_patient_id(doctor_name=doctor_name)

    def list_patients(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for row in self.registry.list_patients():
            detail = self.get_patient(row["patient_id"])
            if detail:
                summaries.append(
                    {
                        "patient_id": detail["patient_id"],
                        "folder": detail["folder"],
                        "document_count": len(detail["documents"]),
                        "complete": detail["complete"],
                        "doc_types": detail["doc_types"],
                        "created_at": detail["created_at"],
                        "updated_at": detail["updated_at"],
                    }
                )
        return summaries

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        row = self.registry.get_patient(patient_id)
        if row is None:
            return None
        documents = storage.list_documents(patient_id)
        doc_types = sorted({doc["doc_type"] for doc in documents})
        return {
            "patient_id": patient_id,
            "folder": row["folder"],
            "doctor_name": row.get("doctor_name"),
            "documents": documents,
            "complete": _REQUIRED_TYPES <= set(doc_types),
            "doc_types": doc_types,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upload_document(
        self,
        patient_id: str,
        doc_type: DocType,
        filename: str,
        data: bytes,
        *,
        doctor_name: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        if self.registry.get_patient(patient_id) is None:
            raise DischargeFlowError(f"Unknown patient {patient_id}. Create the patient first.")
        if doc_type == "discharge_report" and doctor_name:
            self.registry.touch(patient_id, doctor_name=doctor_name)
        result = storage.save_upload(
            patient_id,
            doc_type,
            filename,
            data,
            doctor_name=doctor_name,
            replace=replace,
        )
        self.registry.touch(patient_id)
        return result

    def delete_document(self, patient_id: str, filename: str) -> None:
        if self.registry.get_patient(patient_id) is None:
            raise DischargeFlowError(f"Unknown patient {patient_id}.")
        storage.delete_document(patient_id, filename)
        self.registry.touch(patient_id)

    def process_patient(self, patient_id: str) -> dict[str, Any]:
        detail = self.get_patient(patient_id)
        if detail is None:
            raise DischargeFlowError(f"Unknown patient {patient_id}.")
        if not detail["complete"]:
            missing = sorted(_REQUIRED_TYPES - set(detail["doc_types"]))
            labels = ", ".join(t.replace("_", " ") for t in missing)
            raise DischargeFlowError(
                f"Cannot process {patient_id}: missing {labels}. Upload all three document types."
            )

        from hospital_ai.ui.service import DashboardService

        outcome = DashboardService().process_patient(patient_id)
        _log.info(
            "uploaded patient processed",
            extra={"patient_id": patient_id, "case_id": outcome.get("case_id")},
        )
        return {
            "patient_id": patient_id,
            "case_id": outcome["case_id"],
            "status": outcome["status"],
            "requires_hitl": outcome.get("requires_hitl", False),
            "risk_level": outcome.get("risk_level"),
        }
