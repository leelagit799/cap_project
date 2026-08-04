"""Mock EHR System — FastAPI, port 8050 (doc Table 15).

Serves the patient records the Clinical Validation Agent cross-checks discharge
documents against: demographics, allergies, medication orders, labs and care
plans, plus the simulated clinical-guideline lookup.

No agent imports ``mock_ehr.data`` directly; every read goes through this REST
API, which is what the specification's "cross-checks discharge data against
Mock EHR REST API" (Table 7) requires.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import configure_logging, get_logger
from hospital_ai.ehr.repository import EHRRepository, get_repository

_log = get_logger(__name__, service="mock-ehr")


# --- Response models ---------------------------------------------------------


class Patient(BaseModel):
    patient_id: str
    patient_name: str
    dob: str
    sex: str
    primary_dx: list[str] = Field(default_factory=list)
    service_line: str


class MedicationOrder(BaseModel):
    name: str
    dose: str
    frequency: str


class LabResult(BaseModel):
    test: str
    value: str
    abnormal: bool
    action_in_ehr: str = ""


class CarePlan(BaseModel):
    followup_required: bool
    speciality: str
    window_days: int


class Guideline(BaseModel):
    diagnosis: str
    required_followup: str
    essential_meds: list[str] = Field(default_factory=list)


class AllergyResponse(BaseModel):
    patient_id: str
    allergies: list[str]


class PatientBundle(BaseModel):
    """Everything the Validation Agent needs, in one call."""

    patient: Patient
    allergies: list[str]
    medications: list[MedicationOrder]
    labs: list[LabResult]
    care_plan: CarePlan | None = None
    guidelines: dict[str, Guideline] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    patients: int


# --- Application -------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.reports_dir / "pipeline.log")

    app = FastAPI(
        title="Mock EHR System",
        description=(
            "Simulated hospital EHR for DischargeFlow. Provides patients, "
            "medication orders, allergies, labs and care plans for discharge "
            "cross-validation."
        ),
        version="1.0.0",
    )

    def repo() -> EHRRepository:
        return get_repository()

    def _require_patient(patient_id: str, repository: EHRRepository) -> dict[str, Any]:
        patient = repository.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        return patient

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(repository: EHRRepository = Depends(repo)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="mock-ehr",
            patients=len(repository.list_patients()),
        )

    @app.get("/patients", response_model=list[Patient], tags=["patients"])
    def list_patients(
        service_line: str | None = Query(default=None),
        repository: EHRRepository = Depends(repo),
    ) -> list[dict[str, Any]]:
        return repository.list_patients(service_line)

    @app.get("/patients/{patient_id}", response_model=Patient, tags=["patients"])
    def get_patient(
        patient_id: str, repository: EHRRepository = Depends(repo)
    ) -> dict[str, Any]:
        return _require_patient(patient_id, repository)

    @app.get(
        "/patients/{patient_id}/allergies",
        response_model=AllergyResponse,
        tags=["clinical"],
    )
    def get_allergies(
        patient_id: str, repository: EHRRepository = Depends(repo)
    ) -> AllergyResponse:
        _require_patient(patient_id, repository)
        return AllergyResponse(
            patient_id=patient_id, allergies=repository.get_allergies(patient_id)
        )

    @app.get(
        "/patients/{patient_id}/medications",
        response_model=list[MedicationOrder],
        tags=["clinical"],
    )
    def get_medications(
        patient_id: str, repository: EHRRepository = Depends(repo)
    ) -> list[dict[str, Any]]:
        _require_patient(patient_id, repository)
        return repository.get_medications(patient_id)

    @app.get(
        "/patients/{patient_id}/labs", response_model=list[LabResult], tags=["clinical"]
    )
    def get_labs(
        patient_id: str,
        abnormal_only: bool = Query(default=False),
        repository: EHRRepository = Depends(repo),
    ) -> list[dict[str, Any]]:
        _require_patient(patient_id, repository)
        return (
            repository.get_abnormal_labs(patient_id)
            if abnormal_only
            else repository.get_labs(patient_id)
        )

    @app.get(
        "/patients/{patient_id}/care-plan", response_model=CarePlan, tags=["clinical"]
    )
    def get_care_plan(
        patient_id: str, repository: EHRRepository = Depends(repo)
    ) -> dict[str, Any]:
        _require_patient(patient_id, repository)
        care_plan = repository.get_care_plan(patient_id)
        if care_plan is None:
            raise HTTPException(
                status_code=404, detail=f"No care plan on file for {patient_id}"
            )
        return care_plan

    @app.get("/patients/{patient_id}/bundle", response_model=PatientBundle, tags=["clinical"])
    def get_bundle(
        patient_id: str, repository: EHRRepository = Depends(repo)
    ) -> dict[str, Any]:
        bundle = repository.get_bundle(patient_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        return bundle

    @app.get("/guidelines", response_model=dict[str, Guideline], tags=["guidelines"])
    def list_guidelines(
        repository: EHRRepository = Depends(repo),
    ) -> dict[str, Any]:
        return repository.list_guidelines()

    @app.get("/guidelines/{icd10}", response_model=Guideline, tags=["guidelines"])
    def get_guideline(
        icd10: str, repository: EHRRepository = Depends(repo)
    ) -> dict[str, Any]:
        guideline = repository.get_guideline(icd10)
        if guideline is None:
            raise HTTPException(status_code=404, detail=f"No guideline for ICD-10 {icd10}")
        return guideline

    _log.info("mock EHR application constructed")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "hospital_ai.ehr.app:app",
        host="0.0.0.0",
        port=settings.ports.ehr,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
