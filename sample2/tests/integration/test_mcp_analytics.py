"""P3 — Secondary MCP Analytics Server (:8201) and the shared risk engine."""

from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from hospital_ai.analytics.risk import assess
from hospital_ai.core.schemas import Recommendation, RiskLevel, Severity
from hospital_ai.mcp_servers.analytics.server import create_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def finding(rule_id: str, weight: int, severity=Severity.WARNING, blocking=False, resolved=False):
    return {
        "rule_id": rule_id,
        "severity": severity.value,
        "message": rule_id,
        "weight": weight,
        "blocking": blocking,
        "resolved": resolved,
    }


class TestRiskEngine:
    def test_clean_case_is_low_and_auto_approves(self):
        result = assess([])
        assert result.level is RiskLevel.LOW
        assert result.recommendation is Recommendation.APPROVE
        assert result.discharge_blocked is False

    def test_soft_demographic_gap_stays_low(self):
        """P1020's only gap is a missing address, weight 1 — still auto-approve."""
        result = assess([finding("missing_field.discharge_report.address", 1)])
        assert result.score == 1
        assert result.level is RiskLevel.LOW

    def test_medium_band(self):
        result = assess([finding("med_omission_check", 3), finding("diagnosis_mismatch_check", 4)])
        assert result.score == 7
        assert result.level is RiskLevel.MEDIUM
        assert result.recommendation is Recommendation.EDIT

    def test_allergy_contradiction_always_escalates_to_high(self):
        """Weight 8 lands inside the Medium band, but the hard guardrail wins."""
        result = assess(
            [finding("allergy_contradiction_check", 8, Severity.CRITICAL, blocking=True)]
        )
        assert result.score == 8
        assert result.level is RiskLevel.HIGH
        assert result.discharge_blocked is True
        assert "allergy_contradiction" in result.triggered_guardrails
        assert result.recommendation is Recommendation.REJECT

    def test_low_translation_confidence_is_a_hard_guardrail(self):
        result = assess([], translation_confidence=0.55)
        assert "translation_confidence_below_threshold" in result.triggered_guardrails
        assert result.level is RiskLevel.HIGH

    def test_confidence_above_threshold_does_not_trip(self):
        result = assess([], translation_confidence=0.85)
        assert result.triggered_guardrails == []
        assert result.level is RiskLevel.LOW

    def test_pediatric_service_line_always_requires_hitl(self):
        result = assess([], service_line="Pediatrics")
        assert "service_line_pediatric" in result.triggered_guardrails
        assert result.discharge_blocked is True

    def test_resolved_findings_do_not_score(self):
        result = assess([finding("med_omission_check", 3, resolved=True)])
        assert result.score == 0
        assert result.level is RiskLevel.LOW

    def test_sla_follows_the_risk_tier(self):
        assert assess([]).sla_seconds == 60
        assert assess([finding("med_omission_check", 3)]).sla_seconds == 14400
        assert assess(
            [finding("allergy_contradiction_check", 8, Severity.CRITICAL, blocking=True)]
        ).sla_seconds == 1800


class TestAnalyticsServer:
    async def test_exposes_exactly_the_three_table_8_tools(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            names = {t.name for t in (await client.list_tools()).tools}
        assert names == {
            "calculate_risk_score",
            "get_population_benchmarks",
            "generate_risk_heatmap",
        }

    async def test_calculate_risk_score(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.call_tool(
                "calculate_risk_score",
                {
                    "findings": [
                        finding("allergy_contradiction_check", 8, Severity.CRITICAL, blocking=True)
                    ],
                    "translation_confidence": 0.42,
                },
            )
            payload = json.loads(result.content[0].text)

        assert payload["risk_level"] == "High"
        assert payload["discharge_blocked"] is True
        assert set(payload["triggered_guardrails"]) >= {
            "allergy_contradiction",
            "translation_confidence_below_threshold",
        }
        assert payload["thresholds"] == {"low_max": 2, "medium_max": 8}

    async def test_population_benchmarks_for_selected_codes(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.call_tool(
                "get_population_benchmarks", {"icd10_codes": ["E11.9", "J18.9", "Z99.9"]}
            )
            payload = json.loads(result.content[0].text)

        assert set(payload["benchmarks"]) == {"E11.9", "J18.9"}
        assert payload["unknown_codes"] == ["Z99.9"]
        assert 0 < payload["mean_readmission_30d_rate"] < 1

    async def test_population_benchmarks_default_to_the_whole_cohort(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.call_tool("get_population_benchmarks", {})
            payload = json.loads(result.content[0].text)
        assert payload["cohort_size"] == 10

    async def test_risk_heatmap(self):
        cases = [
            {"service_line": "General Medicine", "risk_level": "Low", "discharge_blocked": False,
             "findings": [finding("missing_field.discharge_report.address", 1)]},
            {"service_line": "General Medicine", "risk_level": "High", "discharge_blocked": True,
             "findings": [finding("allergy_contradiction_check", 8)]},
            {"service_line": "Cardiology", "risk_level": "Medium", "discharge_blocked": False,
             "findings": [finding("med_omission_check", 3)]},
        ]
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.call_tool("generate_risk_heatmap", {"cases": cases})
            payload = json.loads(result.content[0].text)

        assert payload["total_cases"] == 3
        assert payload["blocked_cases"] == 1
        assert payload["block_rate"] == 0.333
        assert payload["matrix"]["General Medicine"] == {"Low": 1, "Medium": 0, "High": 1}
        assert payload["matrix"]["Cardiology"]["Medium"] == 1
        assert len(payload["cells"]) == 6
        assert payload["top_rules"][0]["count"] == 1
