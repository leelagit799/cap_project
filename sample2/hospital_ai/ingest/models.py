"""Request/response models for the upload API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["discharge_report", "lab_report", "bill"]


class PatientCreated(BaseModel):
    patient_id: str
    folder: str


class DocumentInfo(BaseModel):
    filename: str
    doc_type: DocType
    uri: str
    size_bytes: int
    media_type: str


class PatientDetail(BaseModel):
    patient_id: str
    folder: str
    doctor_name: str | None = None
    documents: list[DocumentInfo] = Field(default_factory=list)
    complete: bool = False
    doc_types: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class PatientSummary(BaseModel):
    patient_id: str
    folder: str
    document_count: int
    complete: bool
    doc_types: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class UploadResult(BaseModel):
    patient_id: str
    doc_type: DocType
    filename: str
    uri: str
    size_bytes: int
    replaced: bool = False


class BatchSaveResult(BaseModel):
    patient_id: str
    folder: str
    documents: list[UploadResult] = Field(default_factory=list)
    complete: bool = False


class ProcessResult(BaseModel):
    patient_id: str
    case_id: str
    status: str
    requires_hitl: bool
    risk_level: str | None = None
