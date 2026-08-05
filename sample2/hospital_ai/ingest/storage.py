"""Filesystem storage and automatic renaming for uploaded clinical documents."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.core.logging import get_logger
from hospital_ai.documents.loaders import SUPPORTED_SUFFIXES, media_type_for, sha256_of
from hospital_ai.ingest.models import DocType

_log = get_logger(__name__, component="upload-storage")

_DOC_SUFFIXES = {
    "discharge_report": SUPPORTED_SUFFIXES,
    "lab_report": SUPPORTED_SUFFIXES - {".docx"},
    "bill": SUPPORTED_SUFFIXES - {".docx"},
}


def _sanitize_doctor_name(name: str | None) -> str:
    if not name or not name.strip():
        return "doctor"
    cleaned = re.sub(r"[^\w\-]+", "", name.strip().replace(" ", ""))
    return cleaned[:48] or "doctor"


def target_filename(
    patient_id: str,
    doc_type: DocType,
    original_name: str,
    *,
    doctor_name: str | None = None,
) -> str:
    """Build the canonical on-disk name for one uploaded document."""
    suffix = Path(original_name).suffix.lower()
    if doc_type == "discharge_report":
        label = _sanitize_doctor_name(doctor_name)
        return f"{patient_id}_{label}{suffix}"
    if doc_type == "lab_report":
        return f"{patient_id}_labs{suffix}"
    return f"{patient_id}_bill{suffix}"


def patient_folder(patient_id: str) -> Path:
    settings = get_settings()
    return settings.upload_dir / patient_id


def validate_upload(
    doc_type: DocType,
    filename: str,
    size_bytes: int,
) -> None:
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    allowed = _DOC_SUFFIXES.get(doc_type, SUPPORTED_SUFFIXES)
    if suffix not in allowed:
        raise DischargeFlowError(
            f"Unsupported file type {suffix!r} for {doc_type.replace('_', ' ')}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    if size_bytes <= 0:
        raise DischargeFlowError("Uploaded file is empty.")
    if size_bytes > settings.upload_max_bytes:
        limit_mb = settings.upload_max_bytes / (1024 * 1024)
        raise DischargeFlowError(f"File exceeds the {limit_mb:.0f} MiB upload limit.")


def list_documents(patient_id: str) -> list[dict]:
    folder = patient_folder(patient_id)
    if not folder.is_dir():
        return []

    documents: list[dict] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        doc_type = infer_doc_type(path)
        if doc_type is None:
            continue
        documents.append(
            {
                "filename": path.name,
                "doc_type": doc_type,
                "uri": f"{patient_id}/{path.name}",
                "size_bytes": path.stat().st_size,
                "media_type": media_type_for(path),
                "sha256": sha256_of(path),
            }
        )
    return documents


def infer_doc_type(path: Path) -> DocType | None:
    """Classify a file inside a per-patient upload folder."""
    name = path.name.lower()
    if "_labs" in name or name.endswith("_lab" + path.suffix.lower()):
        return "lab_report"
    if "_bill" in name:
        return "bill"
    patient_prefix = f"{path.parent.name}_".lower()
    if name.startswith(patient_prefix) and "_labs" not in name and "_bill" not in name:
        return "discharge_report"
    return None


def save_upload(
    patient_id: str,
    doc_type: DocType,
    original_name: str,
    data: bytes,
    *,
    doctor_name: str | None = None,
    replace: bool = False,
) -> dict:
    validate_upload(doc_type, original_name, len(data))
    folder = patient_folder(patient_id)
    folder.mkdir(parents=True, exist_ok=True)

    filename = target_filename(patient_id, doc_type, original_name, doctor_name=doctor_name)
    destination = folder / filename
    replaced = False

    for existing in list_documents(patient_id):
        if existing["doc_type"] == doc_type:
            if not replace:
                raise DischargeFlowError(
                    f"A {doc_type.replace('_', ' ')} is already on file "
                    f"({existing['filename']}). Delete it or enable replace."
                )
            if existing["filename"] != filename:
                (folder / existing["filename"]).unlink(missing_ok=True)
            replaced = True

    if destination.exists():
        replaced = True

    temp = folder / f".{filename}.uploading"
    temp.write_bytes(data)
    temp.replace(destination)

    _log.info(
        "document stored",
        extra={"patient_id": patient_id, "doc_type": doc_type, "filename": filename},
    )
    return {
        "patient_id": patient_id,
        "doc_type": doc_type,
        "filename": filename,
        "uri": f"{patient_id}/{filename}",
        "size_bytes": destination.stat().st_size,
        "replaced": replaced,
    }


def save_patient_documents(
    patient_id: str,
    documents: dict[DocType, tuple[str, bytes]],
    *,
    doctor_name: str | None = None,
) -> list[dict]:
    """Validate every upload, then persist all documents in one transaction.

  If any document fails validation, nothing is written to disk.
    """
    required = {"discharge_report", "lab_report", "bill"}
    missing = required - set(documents)
    if missing:
        labels = ", ".join(sorted(t.replace("_", " ") for t in missing))
        raise DischargeFlowError(f"Missing required documents: {labels}.")

    for doc_type, (original_name, data) in documents.items():
        validate_upload(doc_type, original_name, len(data))

    folder = patient_folder(patient_id)
    staging = folder / ".batch_save_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    planned: list[tuple[DocType, str, Path]] = []
    try:
        for doc_type, (original_name, data) in documents.items():
            filename = target_filename(
                patient_id, doc_type, original_name, doctor_name=doctor_name
            )
            temp_path = staging / filename
            temp_path.write_bytes(data)
            planned.append((doc_type, filename, temp_path))

        folder.mkdir(parents=True, exist_ok=True)
        for doc_type, filename, temp_path in planned:
            for existing in list_documents(patient_id):
                if existing["doc_type"] == doc_type and existing["filename"] != filename:
                    (folder / existing["filename"]).unlink(missing_ok=True)
            destination = folder / filename
            temp_path.replace(destination)

        results = []
        for doc_type, filename, _ in planned:
            destination = folder / filename
            results.append(
                {
                    "patient_id": patient_id,
                    "doc_type": doc_type,
                    "filename": filename,
                    "uri": f"{patient_id}/{filename}",
                    "size_bytes": destination.stat().st_size,
                    "replaced": True,
                }
            )
        _log.info(
            "patient documents saved",
            extra={"patient_id": patient_id, "count": len(results)},
        )
        return results
    except Exception:
        for _, filename, temp_path in planned:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def delete_document(patient_id: str, filename: str) -> None:
    folder = patient_folder(patient_id)
    path = (folder / filename).resolve()
    if not path.is_file() or path.parent != folder.resolve():
        raise DischargeFlowError(f"Document not found: {filename}")
    path.unlink()


def delete_patient_folder(patient_id: str) -> None:
    folder = patient_folder(patient_id)
    if folder.is_dir():
        shutil.rmtree(folder)
