"""Clinical Data Harvester Tool — Tool (doc Table 7).

Extracts text, tables and structured payloads from clinical documents in every
format the dataset uses: TXT, JSON, PDF, DOCX and scanned PNG (OCR).

Callers pass the root-relative URI the Clinical Watcher Tool returned. The
harvester re-validates it against the live roots, so an authorised URI from an
earlier session cannot be replayed against a different workspace.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from hospital_ai.core.errors import ExtractionError
from hospital_ai.core.logging import get_logger
from hospital_ai.documents.loaders import load_document
from hospital_ai.mcp_servers.primary.roots import resolve_relative, resolve_roots

_log = get_logger(__name__, tool="clinical-harvester")

#: Cap returned text so a pathological document cannot flood an LLM context.
MAX_TEXT_CHARS = 200_000


async def harvest_document(
    ctx,
    uri: str,
    *,
    include_tables: bool = True,
    allow_local_fallback: bool = False,
) -> dict[str, Any]:
    """Extract content from one root-relative document URI."""
    roots = await resolve_roots(ctx, allow_local_fallback=allow_local_fallback)
    path = resolve_relative(uri, roots)

    try:
        content = load_document(path)
    except ExtractionError as exc:
        _log.error("extraction failed", extra={"uri": uri, "error": str(exc)})
        return {
            "uri": uri,
            "ok": False,
            "error": str(exc),
            "text": "",
            "tables": [],
            "structured": None,
        }

    text = content.text
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]

    result: dict[str, Any] = {
        "uri": uri,
        "ok": True,
        "media_type": content.media_type,
        "text": text,
        "char_count": len(content.text),
        "truncated": truncated,
        "page_count": content.page_count,
        "ocr_used": content.ocr_used,
        "ocr_source": content.ocr_source,
        "structured": content.structured,
        "warnings": content.warnings,
    }
    result["tables"] = content.tables if include_tables else []

    _log.info(
        "document harvested",
        extra={
            "uri": uri,
            "chars": len(content.text),
            "tables": len(content.tables),
            "ocr_used": content.ocr_used,
        },
    )
    return result


async def harvest_patient_packet(
    ctx, uris: list[str], *, allow_local_fallback: bool = False
) -> dict[str, Any]:
    """Extract a whole discharge packet in one call."""
    documents = [
        await harvest_document(ctx, uri, allow_local_fallback=allow_local_fallback)
        for uri in uris
    ]
    return {
        "document_count": len(documents),
        "failed": [d["uri"] for d in documents if not d["ok"]],
        "documents": documents,
    }


def register(mcp) -> None:
    @mcp.tool(
        name="clinical_data_harvester",
        description=(
            "Extract text, tables and structured data from a clinical document "
            "(TXT, JSON, PDF, DOCX or scanned image). Takes the root-relative "
            "URI returned by clinical_watcher."
        ),
    )
    async def clinical_data_harvester(
        ctx: Context, uri: str, include_tables: bool = True
    ) -> dict[str, Any]:
        return await harvest_document(ctx, uri, include_tables=include_tables)

    @mcp.tool(
        name="harvest_patient_packet",
        description=(
            "Extract every document in one patient's discharge packet in a "
            "single call. Takes root-relative URIs from clinical_watcher."
        ),
    )
    async def _harvest_packet(ctx: Context, uris: list[str]) -> dict[str, Any]:
        return await harvest_patient_packet(ctx, uris)
