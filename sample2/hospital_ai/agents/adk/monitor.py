"""Discharge Monitor Agent — Google ADK, port 8103 (doc §2.1).

Scans the incoming document folder for new discharge packets. The folder is
discovered through **MCP Roots**: the agent registers the input directory as a
Root URI when it opens its MCP connection, and the Clinical Watcher Tool calls
``ctx.list_roots()`` to learn where it may read. No raw filesystem path is ever
passed as a tool parameter.

Content hashes are remembered so a case is not reprocessed on every scan; a
document only re-triggers when its bytes change, which is what makes the
watcher usable as a poller.
"""

from __future__ import annotations

from typing import Any

from hospital_ai.agents.gateway import ToolGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, agent="discharge-monitor")

ADK_INSTRUCTION = """\
You are the Discharge Monitor for St. Marian Regional Medical Center. Watch the
authorised clinical input workspace for new discharge packets — doctor reports,
lab reports and hospital bills — and report each patient whose documents have
arrived or changed. You may only read through the Roots-authorised watcher
tool; never accept or construct a filesystem path.
"""


def build_adk_agent():
    """Construct the Google ADK agent definition for this role.

    The ADK ``LlmAgent`` carries the agent's identity, instruction and model
    binding. Scanning itself is deterministic and runs through the MCP watcher
    tool, so the agent definition does not need to reason its way to an answer.
    """
    try:
        from google.adk.agents import LlmAgent

        settings = get_settings()
        agent = LlmAgent(
            name="discharge_monitor",
            description="Detects new patient discharge packets in the Roots-scoped workspace.",
            instruction=ADK_INSTRUCTION,
            model=settings.llm.primary_model.replace("bedrock/", ""),
        )
        _log.info("ADK monitor agent constructed")
        return agent
    except Exception as exc:  # noqa: BLE001 - scanning works without the LLM shell
        _log.warning("ADK LlmAgent unavailable", extra={"error": str(exc)})
        return None


class DischargeMonitorAgent:
    def __init__(self, tools: ToolGateway) -> None:
        self.tools = tools
        self.adk_agent = build_adk_agent()
        #: (root-relative uri) -> sha256 of the content last seen.
        self._seen: dict[str, str] = {}

    async def scan(
        self, patient_id: str | None = None, *, only_new: bool = False
    ) -> dict[str, Any]:
        result = await self.tools.call_tool(
            "clinical_watcher", {"patient_id": patient_id} if patient_id else {}
        )

        patients = result.get("patients", [])
        new_patients: list[dict[str, Any]] = []

        for entry in patients:
            changed = [
                document
                for document in entry["documents"]
                if self._seen.get(document["uri"]) != document["sha256"]
            ]
            entry["changed_documents"] = [d["uri"] for d in changed]
            entry["is_new"] = bool(changed)
            if changed:
                new_patients.append(entry)

        _log.info(
            "workspace scanned",
            extra={
                "roots": result.get("roots"),
                "patients": len(patients),
                "with_changes": len(new_patients),
            },
        )
        return {
            "ok": True,
            "roots": result.get("roots", []),
            "patient_count": result.get("patient_count", 0),
            "document_count": result.get("document_count", 0),
            "patients": new_patients if only_new else patients,
            "new_patients": [entry["patient_id"] for entry in new_patients],
        }

    def acknowledge(self, patients: list[dict[str, Any]]) -> None:
        """Record the hashes of documents whose cases have been started."""
        for entry in patients:
            for document in entry.get("documents", []):
                self._seen[document["uri"]] = document["sha256"]

    async def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.scan(
            patient_id=payload.get("patient_id"),
            only_new=bool(payload.get("only_new", False)),
        )
        if payload.get("acknowledge"):
            self.acknowledge(result["patients"])
        return result
