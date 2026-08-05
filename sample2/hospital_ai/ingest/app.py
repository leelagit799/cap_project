"""Patient document upload API — port 8060.

Writes files into ``data/input/<patient_id>/`` so the existing Clinical Watcher
and Host Orchestrator pipeline can discover and process them without changes to
agent business logic.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.core.logging import configure_logging, get_logger
from hospital_ai.ingest.models import (
    DocType,
    PatientCreated,
    PatientDetail,
    PatientSummary,
    ProcessResult,
    UploadResult,
)
from hospital_ai.ingest.service import UploadService

_log = get_logger(__name__, service="upload-api")


def get_upload_service() -> UploadService:
    return UploadService()


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()

    app = FastAPI(
        title="DischargeFlow Upload API",
        description="Dynamic patient document ingestion for the clinical workspace.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "upload-api"}

    @app.post("/patients", response_model=PatientCreated)
    def create_patient(
        doctor_name: str | None = None,
        service: UploadService = Depends(get_upload_service),
    ) -> PatientCreated:
        payload = service.create_patient(doctor_name=doctor_name)
        return PatientCreated(**payload)

    @app.get("/patients", response_model=list[PatientSummary])
    def list_patients(service: UploadService = Depends(get_upload_service)) -> list[PatientSummary]:
        return [PatientSummary(**row) for row in service.list_patients()]

    @app.get("/patients/{patient_id}", response_model=PatientDetail)
    def get_patient(
        patient_id: str,
        service: UploadService = Depends(get_upload_service),
    ) -> PatientDetail:
        detail = service.get_patient(patient_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        return PatientDetail(**detail)

    @app.post("/patients/{patient_id}/upload", response_model=UploadResult)
    async def upload_document(
        patient_id: str,
        doc_type: Annotated[DocType, Query(description="discharge_report | lab_report | bill")],
        file: UploadFile = File(...),
        doctor_name: str | None = None,
        replace: bool = False,
        service: UploadService = Depends(get_upload_service),
    ) -> UploadResult:
        data = await file.read()
        try:
            result = service.upload_document(
                patient_id,
                doc_type,
                file.filename or "upload.bin",
                data,
                doctor_name=doctor_name,
                replace=replace,
            )
        except DischargeFlowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UploadResult(**result)

    @app.delete("/patients/{patient_id}/documents/{filename}")
    def delete_document(
        patient_id: str,
        filename: str,
        service: UploadService = Depends(get_upload_service),
    ) -> dict[str, str]:
        try:
            service.delete_document(patient_id, filename)
        except DischargeFlowError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "filename": filename}

    @app.post("/patients/{patient_id}/process", response_model=ProcessResult)
    def process_patient(
        patient_id: str,
        service: UploadService = Depends(get_upload_service),
    ) -> ProcessResult:
        try:
            result = service.process_patient(patient_id)
        except DischargeFlowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface orchestrator failures
            _log.error("process failed", extra={"patient_id": patient_id, "error": str(exc)})
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ProcessResult(**result)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    configure_logging()
    port = get_settings().ports.ingest
    _log.info("starting upload api", extra={"port": port})
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
