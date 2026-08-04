"""Clinical Extractor Agent — LangGraph, port 8100 (doc §2.2).

A ``StateGraph`` with a ``MemorySaver`` checkpointer that harvests a patient's
discharge packet and produces structured clinical JSON.

Graph shape::

    discover -> harvest -> parse_discharge -> parse_labs -> parse_bill -> assemble

Conditional edges skip a parse node when its document is absent, so a partial
packet still yields a record with an explicit gap rather than failing outright —
the missing document becomes a Validation Agent finding.

Every node is checkpointed under ``thread_id = case_id``, so a case interrupted
mid-extraction resumes at the node it reached instead of re-reading and
re-OCRing every document.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.ids import utc_now_iso
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import ClinicalRecord, DocType, SourceDocument
from hospital_ai.documents import parsers
from hospital_ai.documents.loaders import DocumentContent

_log = get_logger(__name__, agent="clinical-extractor")


def _merge(left: list, right: list) -> list:
    return [*left, *right]


class ExtractorState(TypedDict, total=False):
    """Typed state threaded through the extraction graph."""

    case_id: str
    patient_id: str
    trace_id: str
    prompt: str
    documents: list[dict[str, Any]]
    primary_documents: dict[str, str]
    harvested: dict[str, dict[str, Any]]
    discharge_report: dict[str, Any] | None
    lab_report: dict[str, Any] | None
    bill: dict[str, Any] | None
    languages: dict[str, str]
    record: dict[str, Any] | None
    warnings: Annotated[list[str], _merge]
    errors: Annotated[list[str], _merge]


def _content_from_harvest(payload: dict[str, Any]) -> DocumentContent:
    return DocumentContent(
        text=payload.get("text") or "",
        media_type=payload.get("media_type") or "text/plain",
        tables=payload.get("tables") or [],
        structured=payload.get("structured"),
        page_count=payload.get("page_count") or 1,
        ocr_used=bool(payload.get("ocr_used")),
        ocr_source=payload.get("ocr_source"),
    )


def build_graph(gateway: ToolGateway):
    """Compile the extraction StateGraph."""

    async def discover(state: ExtractorState) -> dict[str, Any]:
        """Locate the patient's documents through the Roots-scoped watcher."""
        patient_id = state["patient_id"]
        result = await gateway.call_tool("clinical_watcher", {"patient_id": patient_id})

        patients = result.get("patients") or []
        if not patients:
            return {
                "documents": [],
                "primary_documents": {},
                "errors": [f"No documents found for {patient_id} inside the declared roots."],
            }

        entry = patients[0]
        _log.info(
            "documents discovered",
            extra={
                "patient_id": patient_id,
                "documents": len(entry["documents"]),
                "doc_types": entry["doc_types"],
            },
        )
        warnings = []
        if not entry.get("complete"):
            missing = {"discharge_report", "lab_report", "bill"} - set(entry["doc_types"])
            warnings.append(f"Incomplete packet; missing {', '.join(sorted(missing))}.")

        return {
            "documents": entry["documents"],
            "primary_documents": entry.get("primary_documents", {}),
            "warnings": warnings,
        }

    async def fetch_prompt(state: ExtractorState) -> dict[str, Any]:
        """Fetch the extraction instructions as an MCP Prompt, never hardcoded."""
        doc_types = ", ".join(sorted(state.get("primary_documents", {}))) or "unknown"
        languages = {
            doc.get("doc_type"): doc.get("language", "unknown")
            for doc in state.get("documents", [])
        }
        prompt = await gateway.get_prompt(
            "discharge-extraction-prompt",
            {"language": "auto-detected", "doc_types": doc_types},
        )
        return {"prompt": prompt, "languages": languages}

    async def harvest(state: ExtractorState) -> dict[str, Any]:
        """Extract raw content for each preferred document."""
        harvested: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        errors: list[str] = []

        for doc_type, uri in (state.get("primary_documents") or {}).items():
            payload = await gateway.call_tool("clinical_data_harvester", {"uri": uri})
            if not payload.get("ok"):
                errors.append(f"{doc_type}: {payload.get('error')}")
                continue
            harvested[doc_type] = payload
            warnings.extend(payload.get("warnings") or [])

        _log.info(
            "packet harvested",
            extra={"case_id": state.get("case_id"), "documents": sorted(harvested)},
        )
        return {"harvested": harvested, "warnings": warnings, "errors": errors}

    def _parse_node(doc_type: str, key: str):
        async def node(state: ExtractorState) -> dict[str, Any]:
            payload = (state.get("harvested") or {}).get(doc_type)
            if payload is None:
                return {key: None}
            try:
                parsed = parsers.parse(doc_type, _content_from_harvest(payload))
            except Exception as exc:  # noqa: BLE001 - one bad document must not kill the case
                _log.error(
                    "parsing failed",
                    extra={"doc_type": doc_type, "error": str(exc)},
                    exc_info=True,
                )
                return {key: None, "errors": [f"{doc_type}: parsing failed ({exc})"]}
            return {key: parsed.model_dump(mode="json")}

        return node

    async def assemble(state: ExtractorState) -> dict[str, Any]:
        """Build the ClinicalRecord the downstream agents consume."""
        documents = [
            SourceDocument(**{k: v for k, v in doc.items() if k in SourceDocument.model_fields})
            for doc in state.get("documents", [])
        ]
        record = ClinicalRecord(
            patient_id=state["patient_id"],
            case_id=state["case_id"],
            discharge_report=state.get("discharge_report"),
            lab_report=state.get("lab_report"),
            bill=state.get("bill"),
            documents=documents,
            extraction_warnings=state.get("warnings", []),
            extracted_at=utc_now_iso(),
        )
        _log.info(
            "clinical record assembled",
            extra={
                "case_id": state["case_id"],
                "has_discharge": record.discharge_report is not None,
                "has_labs": record.lab_report is not None,
                "has_bill": record.bill is not None,
            },
        )
        return {"record": record.model_dump(mode="json")}

    graph = StateGraph(ExtractorState)
    graph.add_node("discover", discover)
    graph.add_node("fetch_prompt", fetch_prompt)
    graph.add_node("harvest", harvest)
    graph.add_node("parse_discharge", _parse_node(DocType.DISCHARGE_REPORT.value, "discharge_report"))
    graph.add_node("parse_labs", _parse_node(DocType.LAB_REPORT.value, "lab_report"))
    graph.add_node("parse_bill", _parse_node(DocType.BILL.value, "bill"))
    graph.add_node("assemble", assemble)

    graph.set_entry_point("discover")

    def has_documents(state: ExtractorState) -> str:
        return "fetch_prompt" if state.get("primary_documents") else "assemble"

    graph.add_conditional_edges(
        "discover", has_documents, {"fetch_prompt": "fetch_prompt", "assemble": "assemble"}
    )
    graph.add_edge("fetch_prompt", "harvest")
    graph.add_edge("harvest", "parse_discharge")
    graph.add_edge("parse_discharge", "parse_labs")
    graph.add_edge("parse_labs", "parse_bill")
    graph.add_edge("parse_bill", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile(checkpointer=MemorySaver())


class ClinicalExtractorAgent:
    """A2A-facing wrapper around the extraction graph."""

    def __init__(self, gateway: ToolGateway) -> None:
        self.gateway = gateway
        self.graph = build_graph(gateway)

    async def extract(
        self, patient_id: str, case_id: str, trace_id: str = ""
    ) -> dict[str, Any]:
        state: ExtractorState = {
            "patient_id": patient_id,
            "case_id": case_id,
            "trace_id": trace_id,
            "warnings": [],
            "errors": [],
        }
        # thread_id = case_id makes every checkpoint resumable per case.
        config = {"configurable": {"thread_id": case_id}}
        final = await self.graph.ainvoke(state, config=config)

        return {
            "ok": final.get("record") is not None and not final.get("errors"),
            "case_id": case_id,
            "patient_id": patient_id,
            "record": final.get("record"),
            "warnings": final.get("warnings", []),
            "errors": final.get("errors", []),
        }

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A2A entry point."""
        metadata = payload.get("_metadata") or {}
        patient_id = payload.get("patient_id")
        if not patient_id:
            return {"ok": False, "error": "patient_id is required"}
        return await self.extract(
            patient_id=patient_id,
            case_id=payload.get("case_id") or metadata.get("case_id") or f"CASE-{patient_id}",
            trace_id=payload.get("trace_id") or metadata.get("trace_id", ""),
        )
