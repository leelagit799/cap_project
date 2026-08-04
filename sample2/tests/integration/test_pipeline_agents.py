"""P5–P7 — the LangGraph pipeline: extract, normalize, validate.

Each patient's expected outcome is taken from the inline comments in
``mock_ehr/data.py``, which document what test case each record exercises.
Sampling is served by a deterministic stub so these tests do not depend on a
live Bedrock endpoint; the live path is covered separately.
"""

from __future__ import annotations

import contextlib

import mcp.types as types
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from hospital_ai.agents.gateway import SessionGateway
from hospital_ai.agents.langgraph.extractor import ClinicalExtractorAgent
from hospital_ai.agents.langgraph.normalizer import ClinicalNormalizerAgent
from hospital_ai.agents.langgraph.validator import ClinicalValidationAgent
from hospital_ai.core.config import get_settings
from hospital_ai.core.schemas import RiskLevel
from hospital_ai.mcp_servers.analytics.server import create_server as create_analytics
from hospital_ai.mcp_servers.primary.server import create_server as create_primary

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _roots(context):
    workspace = get_settings().roots.workspace.resolve()
    return types.ListRootsResult(
        roots=[types.Root(uri=AnyUrl(workspace.as_uri()), name="clinical-input")]
    )


async def _stub_sampling(context, params: types.CreateMessageRequestParams):
    """Deterministic stand-in for the agent's LiteLLM client.

    Echoes the source text so translation is observable without asserting on
    model wording, while still exercising the full Sampling round trip.
    """
    source = params.messages[0].content.text
    hint = (
        params.modelPreferences.hints[0].name
        if params.modelPreferences and params.modelPreferences.hints
        else "none"
    )
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=source),
        model=f"bedrock/{hint}",
        stopReason="endTurn",
    )


async def _decline_elicitation(context, params: types.ElicitRequestParams):
    """No reviewer is present in an automated run, so gaps stay open."""
    return types.ElicitResult(action="decline")


@contextlib.asynccontextmanager
async def pipeline():
    """Extractor, Normalizer and Validator over live primary + analytics sessions."""
    async with create_connected_server_and_client_session(
        create_primary()._mcp_server,
        list_roots_callback=_roots,
        sampling_callback=_stub_sampling,
        elicitation_callback=_decline_elicitation,
    ) as primary:
        async with create_connected_server_and_client_session(
            create_analytics()._mcp_server
        ) as analytics:
            gateway = SessionGateway(primary, analytics)
            yield (
                ClinicalExtractorAgent(gateway),
                ClinicalNormalizerAgent(gateway),
                ClinicalValidationAgent(gateway),
            )


async def run_case(patient_id: str):
    async with pipeline() as (extractor, normalizer, validator):
        extracted = await extractor.extract(patient_id, f"CASE-{patient_id}")
        assert extracted["ok"], extracted["errors"]
        normalized = await normalizer.normalize(extracted["record"], f"CASE-{patient_id}")
        result = await validator.validate(
            normalized["record"],
            f"CASE-{patient_id}",
            translation_confidence=normalized["translation_confidence"],
        )
        return extracted, normalized, result


class TestExtraction:
    @pytest.mark.parametrize(
        ("patient_id", "name", "medications"),
        [
            ("P1019", "Thomas Wright", 4),
            ("P1020", "Diego Morales", 4),
            ("P1021", "Rohan Gupta", 3),
            ("P1022", "Daan Bakker", 2),
            ("P1023", "Grace Bennett", 4),
            ("P1024", "Bram de Vries", 2),
        ],
    )
    async def test_every_patient_extracts(self, patient_id, name, medications):
        async with pipeline() as (extractor, _, _):
            result = await extractor.extract(patient_id, f"CASE-{patient_id}")

        assert result["ok"], result["errors"]
        record = result["record"]
        discharge = record["discharge_report"]
        assert discharge["patient_name"] == name
        assert len(discharge["medications"]) == medications
        assert record["lab_report"]["tests"]
        assert record["bill"]["total_amount"] is not None

    @pytest.mark.parametrize(
        "patient_id",
        ["P1019", "P1020", "P1021", "P1022", "P1023", "P1024"],
    )
    async def test_document_patient_ids_match_case(self, patient_id):
        """Each sub-document must carry the same patient id as the case."""
        async with pipeline() as (extractor, _, _):
            record = (await extractor.extract(patient_id, f"CASE-{patient_id}"))["record"]

        assert record["patient_id"] == patient_id
        for section in ("discharge_report", "lab_report", "bill"):
            assert record[section]["patient_id"] == patient_id, section

    async def test_documented_gaps_survive_extraction(self):
        """Extraction must not invent values for the dataset's deliberate gaps."""
        async with pipeline() as (extractor, _, _):
            p1020 = (await extractor.extract("P1020", "C1"))["record"]["discharge_report"]
            p1021 = (await extractor.extract("P1021", "C2"))["record"]["discharge_report"]
            p1024 = (await extractor.extract("P1024", "C3"))["record"]["discharge_report"]

        assert p1020["address"] is None
        assert p1021["address"] is None and p1021["follow_up_appointments"] == []
        assert p1024["age"] is None and p1024["attending_physician"] is None

    async def test_checkpoints_make_a_case_resumable(self):
        async with pipeline() as (extractor, _, _):
            await extractor.extract("P1019", "CASE-RESUME")
            state = await extractor.graph.aget_state(
                {"configurable": {"thread_id": "CASE-RESUME"}}
            )

        assert state.values["record"] is not None
        assert state.values["harvested"].keys() == {"discharge_report", "lab_report", "bill"}


class TestNormalization:
    @pytest.mark.parametrize(
        ("patient_id", "language"),
        [("P1019", "en"), ("P1020", "es"), ("P1021", "hi"), ("P1022", "nl"), ("P1024", "nl")],
    )
    async def test_language_detection(self, patient_id, language):
        async with pipeline() as (extractor, normalizer, _):
            record = (await extractor.extract(patient_id, "C"))["record"]
            result = await normalizer.normalize(record, "C")
        assert result["source_language"] == language

    async def test_english_records_skip_sampling(self):
        async with pipeline() as (extractor, normalizer, _):
            record = (await extractor.extract("P1019", "C"))["record"]
            result = await normalizer.normalize(record, "C")

        assert result["sampling_used"] is False
        assert result["translation_confidence"] == 1.0

    async def test_non_english_records_go_through_sampling(self):
        async with pipeline() as (extractor, normalizer, _):
            record = (await extractor.extract("P1022", "C"))["record"]
            result = await normalizer.normalize(record, "C")

        assert result["sampling_used"] is True
        assert result["model_used"] == "bedrock/nova-lite"

    async def test_abbreviations_expand_without_touching_drug_facts(self):
        """BID becomes twice daily; the drug, strength and quantity do not move."""
        async with pipeline() as (extractor, normalizer, _):
            record = (await extractor.extract("P1019", "C"))["record"]
            result = await normalizer.normalize(record, "C")

        metformin = result["record"]["discharge_report"]["medications"][0]
        assert metformin["medicine_name"] == "Metformin"
        assert metformin["strength"] == "500 mg"
        assert metformin["frequency"] == "BID"
        assert metformin["frequency_expanded"] == "twice daily"


class TestValidationOutcomes:
    """Each expectation comes from the comments in mock_ehr/data.py."""

    @pytest.mark.parametrize("patient_id", ["P1019", "P1023"])
    async def test_clean_cases_auto_approve(self, patient_id):
        _, _, result = await run_case(patient_id)

        assert result.risk_level is RiskLevel.LOW
        assert result.discharge_blocked is False
        assert result.requires_hitl is False
        assert result.recommendation.value == "Approve"
        assert result.completeness_score == 100.0

    @pytest.mark.parametrize("patient_id", ["P1022", "P1024"])
    async def test_allergy_contradiction_hard_blocks(self, patient_id):
        """Penicillin on file, Amoxicilline prescribed."""
        _, _, result = await run_case(patient_id)

        rule_ids = {f.rule_id for f in result.findings}
        assert "allergy_contradiction_check" in rule_ids
        assert result.discharge_blocked is True
        assert result.risk_level is RiskLevel.HIGH
        assert result.requires_hitl is True
        assert "allergy_contradiction" in result.triggered_guardrails

        allergy = next(f for f in result.findings if f.rule_id == "allergy_contradiction_check")
        assert allergy.severity.value == "critical"
        assert "Amoxicilline" in str(allergy.actual)

    async def test_p1024_also_reports_its_missing_demographics(self):
        _, _, result = await run_case("P1024")
        missing = {f.field for f in result.findings if f.rule_id.startswith("missing_field.")}
        assert "age" in missing
        assert "doctors" in missing

    async def test_p1021_escalates_on_bill_and_follow_up(self):
        """Unpaid bill plus missing address and follow-up."""
        _, _, result = await run_case("P1021")

        rule_ids = {f.rule_id for f in result.findings}
        assert "bill_settlement_check" in rule_ids
        assert "follow_up_missing_check" in rule_ids
        assert "missing_field.discharge_report.address" in rule_ids
        assert result.discharge_blocked is True
        assert result.requires_hitl is True

    async def test_p1020_missing_address_alone_stays_low_risk(self):
        """A soft demographic gap is weight 1 and must not block release."""
        _, _, result = await run_case("P1020")

        missing = {f.field for f in result.findings if f.rule_id.startswith("missing_field.")}
        assert missing == {"address"}
        assert result.risk_level is RiskLevel.LOW
        assert result.discharge_blocked is False

    async def test_declined_elicitation_leaves_the_gap_open(self):
        _, _, result = await run_case("P1020")

        assert result.elicitations
        assert result.elicitations[0].action.value == "decline"
        assert "address" in result.elicitations[0].fields_requested
        address = next(f for f in result.findings if f.field == "address")
        assert address.resolved is False

    async def test_medications_reconcile_across_languages(self):
        """Metformina and Atorvastatina must match the EHR's Spanish orders."""
        _, _, result = await run_case("P1020")
        assert "med_omission_check" not in {f.rule_id for f in result.findings}

    async def test_every_result_carries_the_rules_version(self):
        _, _, result = await run_case("P1019")
        assert result.rules_version == get_settings().rules_version
        assert len(result.rules_version) == 64
