"""P3 — live dual-server MCP connectivity over streamable-HTTP.

Doc §4 requires agents to hold sessions with both MCP servers at once. These
tests exercise the real transport against running servers, so they are skipped
unless :8200, :8201 and :8050 are up.

Start them with::

    python -m hospital_ai.ehr.app
    python -m hospital_ai.mcp_servers.primary.server
    python -m hospital_ai.mcp_servers.analytics.server
"""

from __future__ import annotations

import socket

import mcp.types as types
import pytest

from hospital_ai.core.config import get_settings
from hospital_ai.mcp_servers.client import MultiServerMCPClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


_ports = get_settings().ports
requires_servers = pytest.mark.skipif(
    not (_listening(_ports.primary_mcp) and _listening(_ports.analytics_mcp)),
    reason="MCP servers are not running on :8200 and :8201",
)
requires_ehr = pytest.mark.skipif(
    not _listening(_ports.ehr), reason="Mock EHR is not running on :8050"
)


async def _echo_sampler(params: types.CreateMessageRequestParams) -> types.CreateMessageResult:
    hint = params.modelPreferences.hints[0].name if params.modelPreferences and params.modelPreferences.hints else "none"
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=f"[translated via {hint}] discharge summary"),
        model=f"bedrock/{hint}",
        stopReason="endTurn",
    )


@requires_servers
class TestDualServerSessions:
    async def test_client_holds_both_sessions_simultaneously(self):
        async with MultiServerMCPClient() as client:
            tools = await client.list_all_tools()

        assert set(tools) == {"clinical-tools", "analytics-tools"}
        assert "clinical_watcher" in tools["clinical-tools"]
        assert "calculate_risk_score" in tools["analytics-tools"]

    async def test_calls_route_to_the_right_server(self):
        async with MultiServerMCPClient() as client:
            watcher = await client.call_tool("clinical_watcher", {})
            risk = await client.call_tool(
                "calculate_risk_score",
                {"findings": [{"rule_id": "med_omission_check", "weight": 3, "severity": "warning"}]},
            )

        assert watcher["patient_count"] == 6
        assert risk["risk_level"] == "Medium"

    async def test_roots_are_negotiated_over_http(self):
        async with MultiServerMCPClient() as client:
            result = await client.call_tool("clinical_watcher", {"patient_id": "P1023"})

        assert result["patient_count"] == 1
        uris = result["patients"][0]["primary_documents"]
        assert uris["discharge_report"] == "doctor_reports/P1023_grace_bennett.png"

    async def test_sampling_round_trips_over_http(self):
        async with MultiServerMCPClient(sampling_handler=_echo_sampler) as client:
            result = await client.call_tool(
                "medical_lang_bridge",
                {"text": "El paciente fue dado de alta del hospital con diagnóstico de neumonía."},
            )

        assert result["source_language"] == "es"
        assert result["sampling_used"] is True
        assert result["model_hints"] == ["nova-lite"]
        assert "translated via nova-lite" in result["translated_text"]

    async def test_resources_and_prompts_over_http(self):
        async with MultiServerMCPClient() as client:
            rules = await client.read_resource("resource://clinical-rules/cross-validation")
            prompt = await client.get_prompt(
                "summary-generation-prompt", {"risk_level": "Low", "audience": "patient"}
            )

        assert "risk_scoring_matrix" in rules
        assert "Low risk case" in prompt

    @requires_ehr
    async def test_cross_validation_reaches_the_mock_ehr(self):
        """P1022: Penicillin on file, Amoxicilline prescribed — hard block."""
        packet = {
            "discharge_report": {
                "patient_id": "P1022",
                "medications": [
                    {"medicine_name": "Amoxicilline", "strength": "500 mg",
                     "frequency": "TID", "route": "ORAL"},
                    {"medicine_name": "Paracetamol", "strength": "500 mg",
                     "frequency": "q6h PRN", "route": "ORAL"},
                ],
                "follow_up_appointments": ["Huisarts 2026-06-10"],
                "discharge_approved": True,
                "discharge_approved_by": "Dr. van Dijk",
                "icd10_codes": ["J18.9"],
            },
            "bill": {"payment_status": "PAID", "total_amount": 1400.0},
        }
        async with MultiServerMCPClient() as client:
            result = await client.call_tool(
                "ehr_validation", {"patient_id": "P1022", "packet": packet}
            )

        rule_ids = [f["rule_id"] for f in result["findings"]]
        assert "allergy_contradiction_check" in rule_ids

        allergy = next(f for f in result["findings"] if f["rule_id"] == "allergy_contradiction_check")
        assert allergy["severity"] == "critical"
        assert allergy["blocking"] is True
        # Amoxicilline -> amoxicillin and Paracetamol -> acetaminophen both
        # reconcile against the EHR, so the allergy is the only med finding.
        assert "med_omission_check" not in rule_ids

    @requires_ehr
    async def test_clean_case_produces_no_blocking_findings(self):
        """P1019 is the fully reconciled auto-approve case."""
        packet = {
            "discharge_report": {
                "patient_id": "P1019",
                "medications": [
                    {"medicine_name": n, "strength": s, "frequency": f, "route": "ORAL"}
                    for n, s, f in [
                        ("Metformin", "500 mg", "BID"), ("Lisinopril", "10 mg", "QD"),
                        ("Atorvastatin", "20 mg", "QHS"), ("Aspirin", "81 mg", "QD"),
                    ]
                ],
                "follow_up_appointments": ["Endocrinology 2026-06-12", "PCP 2026-06-05"],
                "discharge_instructions": "Continue diabetic diet. Recheck HbA1c and LDL.",
                "discharge_approved": True,
                "discharge_approved_by": "Dr. Rachel Greene",
                "icd10_codes": ["E11.9", "I10"],
            },
            "bill": {"payment_status": "PAID", "total_amount": 1903.07},
        }
        async with MultiServerMCPClient() as client:
            validation = await client.call_tool(
                "ehr_validation", {"patient_id": "P1019", "packet": packet}
            )
            risk = await client.call_tool(
                "calculate_risk_score", {"findings": validation["findings"]}
            )

        assert [f for f in validation["findings"] if f["blocking"]] == []
        assert risk["risk_level"] == "Low"
        assert risk["discharge_blocked"] is False
        assert risk["recommendation"] == "Approve"

    @requires_ehr
    async def test_full_chain_produces_an_audit_report(self):
        """Watcher -> harvester -> validation -> risk -> reporter, all over MCP."""
        packet = {
            "discharge_report": {
                "patient_id": "P1021",
                "medications": [
                    {"medicine_name": "Metformin", "strength": "500 mg",
                     "frequency": "BID", "route": "ORAL"},
                    {"medicine_name": "Amlodipine", "strength": "5 mg",
                     "frequency": "QD", "route": "ORAL"},
                    {"medicine_name": "Atorvastatin", "strength": "20 mg",
                     "frequency": "QHS", "route": "ORAL"},
                ],
                "follow_up_appointments": [],
                "discharge_approved": True,
                "discharge_approved_by": "Dr. Anjali Mehta",
                "icd10_codes": ["E11.9", "I10"],
            },
            "bill": {"payment_status": "UNPAID", "total_amount": 1200.0, "currency": "INR"},
        }
        async with MultiServerMCPClient() as client:
            validation = await client.call_tool(
                "ehr_validation", {"patient_id": "P1021", "packet": packet}
            )
            risk = await client.call_tool(
                "calculate_risk_score",
                {"findings": validation["findings"], "translation_confidence": 0.61},
            )
            report = await client.call_tool(
                "clinical_insight_reporter",
                {
                    "case_id": "CASE-P1021-TEST",
                    "patient_id": "P1021",
                    "trace_id": "trace-test-p1021",
                    "validation": {**validation, **risk},
                    "patient_name": "Rohan Gupta",
                    "bill": packet["bill"],
                    "translation_confidence": 0.61,
                },
            )

        rule_ids = {f["rule_id"] for f in validation["findings"]}
        assert "bill_settlement_check" in rule_ids
        assert "follow_up_missing_check" in rule_ids

        assert report["risk_level"] == "High"
        assert report["discharge_blocked"] is True
        assert report["artifacts"]["json"].endswith("audit.json")
        assert report["artifacts"]["html"].endswith("audit.html")
        assert len(report["rules_version"]) == 64

        from pathlib import Path

        html = Path(report["artifacts"]["html"]).read_text(encoding="utf-8")
        assert "Discharge blocked" in html
        assert "Rohan Gupta" in html
