"""Clinical Watcher Tool — Tool + Roots (doc Table 7, Table 9).

Detects new discharge files inside the Roots-scoped workspace. The tool takes
no path parameter of any kind: it calls ``ctx.list_roots()`` to learn where it
is allowed to look, and validates every candidate with ``Path.relative_to()``
before returning it.

Returned URIs are root-relative, so a caller can only ever name a file the
Watcher already authorised.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context

from hospital_ai.core.config import get_settings
from hospital_ai.core.ids import extract_patient_id, PATIENT_ID_PATTERN
from hospital_ai.core.logging import get_logger
from hospital_ai.documents.loaders import (
    ENCODING_PRIORITY,
    SUPPORTED_SUFFIXES,
    find_ocr_sidecar,
    is_ocr_sidecar,
    media_type_for,
    sha256_of,
)
from hospital_ai.mcp_servers.primary.roots import (
    ensure_within_roots,
    relative_uri,
    resolve_roots,
)

_log = get_logger(__name__, tool="clinical-watcher")


def _by_patient_folder_doctype(path: Path) -> str | None:
    """Classify files in ``data/input/P1025/P1025_DrSmith.pdf`` layouts."""
    parent = path.parent.name
    if not PATIENT_ID_PATTERN.fullmatch(parent):
        return None
    name = path.stem.lower()
    if "_labs" in name or name.endswith("_lab"):
        return "lab_report"
    if "_bill" in name:
        return "bill"
    if extract_patient_id(path.name) and path.name.startswith(f"{parent}_"):
        return "discharge_report"
    return None


def _doctype_of(path: Path, roots: list[Path]) -> str | None:
    """Infer the document type from the folder the file sits in."""
    settings = get_settings()
    folder_to_doctype = {v: k for k, v in settings.roots.doctype_dirs.items()}

    for part in reversed(path.parts):
        if part in folder_to_doctype:
            return folder_to_doctype[part]

    by_patient = _by_patient_folder_doctype(path)
    if by_patient:
        return by_patient

    # `by_patient` layout: Data/incoming/P1019/labs.txt
    name = path.name.lower()
    for doctype, keywords in (
        ("lab_report", ("lab", "labs")),
        ("bill", ("bill", "invoice")),
        ("discharge_report", ("discharge", "report", "summary")),
    ):
        if any(keyword in name for keyword in keywords):
            return doctype
    return None


def _describe(path: Path, roots: list[Path]) -> dict[str, Any] | None:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES or is_ocr_sidecar(path):
        return None

    patient_id = extract_patient_id(path.name)
    if patient_id is None:
        _log.warning("file has no patient prefix; quarantined", extra={"file": path.name})
        return None

    doctype = _doctype_of(path, roots)
    if doctype is None:
        _log.warning("could not classify document", extra={"file": path.name})
        return None

    stat = path.stat()
    sidecar = find_ocr_sidecar(path)
    return {
        "patient_id": patient_id,
        "doc_type": doctype,
        "uri": relative_uri(path, roots),
        "media_type": media_type_for(path),
        "sha256": sha256_of(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
        "ocr_sidecar": relative_uri(sidecar, roots) if sidecar else None,
    }


async def scan_clinical_workspace(
    ctx,
    patient_id: str | None = None,
    *,
    allow_local_fallback: bool = False,
) -> dict[str, Any]:
    """Discover discharge packets inside the authorised Roots workspace.

    ``patient_id`` filters the result; it is not a path and cannot widen the
    search beyond the declared roots.
    """
    roots = await resolve_roots(ctx, allow_local_fallback=allow_local_fallback)

    documents: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            # Re-validate every candidate even though rglob started at the root:
            # a symlink inside the tree can still point outside it.
            try:
                ensure_within_roots(path, roots)
            except Exception:  # noqa: BLE001 - rejection is the expected outcome
                continue
            described = _describe(path, roots)
            if described is not None:
                documents.append(described)

    if patient_id:
        documents = [d for d in documents if d["patient_id"] == patient_id]

    patients: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        patients.setdefault(document["patient_id"], []).append(document)

    def preferred(docs: list[dict[str, Any]]) -> dict[str, str]:
        """One document per type, preferring machine-readable encodings."""
        chosen: dict[str, dict[str, Any]] = {}
        for doc in docs:
            rank = ENCODING_PRIORITY.get(Path(doc["uri"]).suffix.lower(), 9)
            current = chosen.get(doc["doc_type"])
            if current is None or rank < current["_rank"]:
                chosen[doc["doc_type"]] = {**doc, "_rank": rank}
        return {doc_type: doc["uri"] for doc_type, doc in chosen.items()}

    _log.info(
        "workspace scanned",
        extra={
            "roots": [str(r) for r in roots],
            "patients": len(patients),
            "documents": len(documents),
        },
    )
    return {
        "roots": [root.as_uri() for root in roots],
        "patient_count": len(patients),
        "document_count": len(documents),
        "patients": [
            {
                "patient_id": pid,
                "documents": sorted(docs, key=lambda d: (d["doc_type"], d["uri"])),
                "primary_documents": preferred(docs),
                "doc_types": sorted({d["doc_type"] for d in docs}),
                "complete": {"discharge_report", "lab_report", "bill"}
                <= {d["doc_type"] for d in docs},
            }
            for pid, docs in sorted(patients.items())
        ],
    }


def register(mcp) -> None:
    @mcp.tool(
        name="clinical_watcher",
        description=(
            "Detect discharge packets inside the Roots-authorised workspace. "
            "Discovers folders via ctx.list_roots(); accepts no filesystem path "
            "and rejects anything outside the declared roots."
        ),
    )
    async def clinical_watcher(ctx: Context, patient_id: str | None = None) -> dict[str, Any]:
        return await scan_clinical_workspace(ctx, patient_id)
