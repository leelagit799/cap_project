"""Filesystem management for shipped static sample patients (P1019–P1024)."""

from __future__ import annotations

import shutil
from pathlib import Path

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.core.ids import extract_patient_id
from hospital_ai.core.logging import get_logger
from hospital_ai.documents.loaders import (
    ENCODING_PRIORITY,
    SUPPORTED_SUFFIXES,
    find_ocr_sidecar,
    is_ocr_sidecar,
    media_type_for,
    sha256_of,
)
from hospital_ai.ingest.models import DocType
from hospital_ai.ingest.storage import target_filename, validate_upload

_log = get_logger(__name__, component="static-samples")

SAMPLE_PATIENT_IDS: tuple[str, ...] = tuple(f"P{n:04d}" for n in range(1019, 1025))
_REQUIRED_TYPES: set[DocType] = {"discharge_report", "lab_report", "bill"}


def list_sample_patients() -> list[str]:
    """Return the fixed static sample patient ids."""
    return list(SAMPLE_PATIENT_IDS)


def _folder_for_doctype(doc_type: DocType) -> Path:
    settings = get_settings()
    subdir = settings.roots.doctype_dirs[doc_type]
    return settings.roots.workspace / subdir


def _classify_by_folder(path: Path) -> DocType | None:
    folder_to_doctype = {v: k for k, v in get_settings().roots.doctype_dirs.items()}
    return folder_to_doctype.get(path.parent.name)  # type: ignore[return-value]


def _scan_patient_files(patient_id: str) -> list[Path]:
    settings = get_settings()
    workspace = settings.roots.workspace
    if not workspace.is_dir():
        return []

    matches: list[Path] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or is_ocr_sidecar(path):
            continue
        if extract_patient_id(path.name) != patient_id:
            continue
        if _classify_by_folder(path) is None:
            continue
        matches.append(path)
    return matches


def _document_row(path: Path, doc_type: DocType, patient_id: str) -> dict:
    rel = path.relative_to(get_settings().roots.workspace)
    return {
        "filename": path.name,
        "doc_type": doc_type,
        "uri": str(rel).replace("\\", "/"),
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type_for(path),
        "sha256": sha256_of(path),
    }


def list_documents(patient_id: str) -> list[dict]:
    """List every on-disk document for a static sample patient."""
    if patient_id not in SAMPLE_PATIENT_IDS:
        return []

    by_type: dict[DocType, list[dict]] = {doc_type: [] for doc_type in _REQUIRED_TYPES}
    for path in _scan_patient_files(patient_id):
        doc_type = _classify_by_folder(path)
        if doc_type is None:
            continue
        by_type[doc_type].append(_document_row(path, doc_type, patient_id))

    documents: list[dict] = []
    for doc_type in sorted(_REQUIRED_TYPES):
        documents.extend(by_type[doc_type])
    return documents


def _primary_documents(documents: list[dict]) -> dict[DocType, dict]:
    """Pick one file per doc type, preferring machine-readable encodings."""
    chosen: dict[DocType, dict] = {}
    for doc in documents:
        doc_type = doc["doc_type"]
        rank = ENCODING_PRIORITY.get(Path(doc["filename"]).suffix.lower(), 9)
        current = chosen.get(doc_type)
        if current is None or rank < current["_rank"]:
            chosen[doc_type] = {**doc, "_rank": rank}
    return {doc_type: {k: v for k, v in row.items() if k != "_rank"} for doc_type, row in chosen.items()}


def get_patient(patient_id: str) -> dict | None:
    """Return document metadata for one static sample patient."""
    if patient_id not in SAMPLE_PATIENT_IDS:
        return None

    documents = list_documents(patient_id)
    primary = _primary_documents(documents)
    doc_types = sorted(primary)
    settings = get_settings()
    return {
        "patient_id": patient_id,
        "folder": str(settings.roots.workspace),
        "doctor_name": None,
        "documents": list(primary.values()),
        "all_documents": documents,
        "complete": _REQUIRED_TYPES <= set(doc_types),
        "doc_types": doc_types,
        "is_static_sample": True,
    }


def _remove_doc_type_files(patient_id: str, doc_type: DocType) -> None:
    for path in _scan_patient_files(patient_id):
        if _classify_by_folder(path) != doc_type:
            continue
        sidecar = find_ocr_sidecar(path)
        path.unlink(missing_ok=True)
        if sidecar and sidecar.is_file():
            sidecar.unlink(missing_ok=True)


def _target_path(patient_id: str, doc_type: DocType, original_name: str, *, doctor_name: str | None) -> Path:
    filename = target_filename(patient_id, doc_type, original_name, doctor_name=doctor_name)
    return _folder_for_doctype(doc_type) / filename


def update_patient_documents(
    patient_id: str,
    *,
    replacements: dict[DocType, tuple[str, bytes]],
    deleted_types: set[DocType] | None = None,
    doctor_name: str | None = None,
) -> list[dict]:
    """Apply document edits for an existing static sample patient.

    Unmentioned document types are left unchanged. Every type listed in
    ``deleted_types`` must also have a replacement in ``replacements``.
    After the update, all three required document types must be present.
    """
    if patient_id not in SAMPLE_PATIENT_IDS:
        raise DischargeFlowError(f"Unknown static sample patient {patient_id}.")

    deleted = deleted_types or set()
    missing_replacements = deleted - set(replacements)
    if missing_replacements:
        labels = ", ".join(sorted(t.replace("_", " ") for t in missing_replacements))
        raise DischargeFlowError(f"Deleted documents must be replaced before saving: {labels}.")

    current = _primary_documents(list_documents(patient_id))
    final_types: set[DocType] = set()

    for doc_type in _REQUIRED_TYPES:
        if doc_type in replacements:
            final_types.add(doc_type)
        elif doc_type in deleted:
            continue
        elif doc_type in current:
            final_types.add(doc_type)

    if final_types != _REQUIRED_TYPES:
        missing = sorted(_REQUIRED_TYPES - final_types)
        labels = ", ".join(t.replace("_", " ") for t in missing)
        raise DischargeFlowError(
            f"All three documents are required before saving. Missing: {labels}."
        )

    for doc_type, (original_name, data) in replacements.items():
        validate_upload(doc_type, original_name, len(data))

    staging_root = get_settings().roots.workspace / ".patient_doc_staging" / patient_id
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    planned: list[tuple[DocType, Path, Path]] = []
    try:
        for doc_type, (original_name, data) in replacements.items():
            destination = _target_path(
                patient_id, doc_type, original_name, doctor_name=doctor_name
            )
            temp_path = staging_root / destination.name
            temp_path.write_bytes(data)
            planned.append((doc_type, temp_path, destination))

        for doc_type, temp_path, destination in planned:
            _remove_doc_type_files(patient_id, doc_type)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path.replace(destination)

        saved = list_documents(patient_id)
        primary = _primary_documents(saved)
        if set(primary) != _REQUIRED_TYPES:
            raise DischargeFlowError("Save failed: patient packet is incomplete after update.")

        _log.info(
            "static patient documents updated",
            extra={"patient_id": patient_id, "replaced": sorted(replacements)},
        )
        return list(primary.values())
    except Exception:
        for doc_type, temp_path, destination in planned:
            temp_path.unlink(missing_ok=True)
            if destination.is_file() and doc_type in replacements:
                destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
