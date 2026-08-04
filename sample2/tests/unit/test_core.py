"""P0 — foundation: config, identifiers, retry, schemas."""

from __future__ import annotations

import asyncio

import pytest

from hospital_ai.core.errors import DischargeFlowError, TransportError
from hospital_ai.core.ids import extract_patient_id, new_case_id, new_trace_id
from hospital_ai.core.retry import CircuitBreaker, retry_async, retry_sync
from hospital_ai.core.schemas import (
    Bill,
    DischargeReport,
    Finding,
    PaymentStatus,
    Prescription,
    RagTriad,
    RiskLevel,
    Severity,
    ValidationResult,
)


class TestConfig:
    def test_ports_match_table_15(self, settings):
        expected = {
            "ehr": 8050, "extractor": 8100, "validator": 8101, "normalizer": 8102,
            "monitor": 8103, "summary": 8104, "rag": 8105, "host": 8083,
            "primary_mcp": 8200, "analytics_mcp": 8201, "dashboard": 8501,
        }
        for name, port in expected.items():
            assert getattr(settings.ports, name) == port

    def test_rules_version_is_sha256(self, settings):
        assert len(settings.rules_version) == 64
        assert set(settings.rules_version) <= set("0123456789abcdef")

    def test_roots_workspace_exists_and_is_a_file_uri(self, settings):
        assert settings.roots.workspace.is_dir()
        assert settings.roots.uri.startswith("file://")

    def test_rules_expose_the_documented_sections(self, settings):
        for section in (
            "mandatory_clinical_fields",
            "mandatory_prescription_fields",
            "normalization_standards",
            "clinical_validation_policies",
            "risk_scoring_matrix",
            "quality_thresholds",
        ):
            assert section in settings.rules

    def test_prompts_cover_table_2(self, settings):
        for prompt in (
            "discharge-extraction-prompt",
            "ehr-cross-validation-prompt",
            "abbreviation-normalization-prompt",
            "summary-generation-prompt",
            "rag-answer-prompt",
        ):
            assert prompt in settings.prompts

    def test_no_secret_leaks_into_yaml(self, settings):
        import json

        blob = json.dumps({"rules": settings.rules, "prompts": settings.prompts})
        assert settings.agent_auth_token not in blob


class TestIdentifiers:
    def test_case_id_embeds_patient(self):
        assert new_case_id("P1019").startswith("CASE-P1019-")

    def test_trace_ids_are_unique(self):
        assert len({new_trace_id() for _ in range(100)}) == 100

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("P1019_labs.txt", "P1019"),
            ("P1023_grace_bennett.png.ocr.txt", "P1023"),
            ("bills/P1020_bill.pdf", "P1020"),
            ("notes.txt", None),
        ],
    )
    def test_patient_id_extraction(self, filename, expected):
        assert extract_patient_id(filename) == expected


class TestRetry:
    def test_sync_retries_then_succeeds(self):
        calls = {"n": 0}

        @retry_sync(max_attempts=3, base_delay=0.001, max_delay=0.002)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransportError("boom")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_non_retryable_errors_propagate_immediately(self):
        calls = {"n": 0}

        @retry_sync(max_attempts=3, base_delay=0.001)
        def fatal():
            calls["n"] += 1
            raise DischargeFlowError("not retryable")

        with pytest.raises(DischargeFlowError):
            fatal()
        assert calls["n"] == 1

    def test_async_retry(self):
        calls = {"n": 0}

        @retry_async(max_attempts=3, base_delay=0.001, max_delay=0.002)
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise TransportError("boom")
            return "ok"

        assert asyncio.run(flaky()) == "ok"

    def test_circuit_breaker_opens_at_threshold(self):
        breaker = CircuitBreaker("extractor", threshold=3)
        for _ in range(2):
            breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open
        breaker.record_success()
        assert not breaker.is_open


class TestSchemas:
    def test_clinical_fields_are_optional_so_gaps_become_findings(self):
        """A blank report must parse; the validator decides what is missing."""
        report = DischargeReport()
        assert report.patient_id is None
        assert report.medications == []

    def test_doctors_merges_attending_and_consulting(self):
        report = DischargeReport(
            attending_physician="Dr. Greene",
            consulting_doctors=["Dr. Patel"],
        )
        assert report.doctors == ["Dr. Greene", "Dr. Patel"]

    def test_prescription_carries_all_nine_columns(self, settings):
        for field in settings.rules["mandatory_prescription_fields"]:
            assert field in Prescription.model_fields

    @pytest.mark.parametrize(
        ("status", "settled"),
        [
            (PaymentStatus.PAID, True),
            (PaymentStatus.INSURANCE_GUARANTEED, True),
            (PaymentStatus.UNPAID, False),
            (PaymentStatus.PARTIAL, False),
            (PaymentStatus.UNKNOWN, False),
        ],
    )
    def test_bill_settlement_rule(self, status, settled):
        assert Bill(payment_status=status).is_settled is settled

    def test_hitl_gate_trips_on_block_or_high_risk(self):
        clean = ValidationResult(case_id="c", patient_id="P1019")
        assert not clean.requires_hitl

        blocked = ValidationResult(case_id="c", patient_id="P1022", discharge_blocked=True)
        assert blocked.requires_hitl

        high = ValidationResult(case_id="c", patient_id="P1024", risk_level=RiskLevel.HIGH)
        assert high.requires_hitl

    def test_blocking_findings_exclude_resolved_ones(self):
        result = ValidationResult(
            case_id="c",
            patient_id="P1021",
            findings=[
                Finding(rule_id="missing_address", severity=Severity.WARNING,
                        message="no address", blocking=True, resolved=True),
                Finding(rule_id="allergy_contradiction_check", severity=Severity.CRITICAL,
                        message="penicillin", blocking=True),
            ],
        )
        assert [f.rule_id for f in result.blocking_findings] == ["allergy_contradiction_check"]
        assert len(result.critical_findings) == 1

    def test_rag_triad_faithfulness_threshold(self):
        assert RagTriad(faithfulness=0.7).passes
        assert not RagTriad(faithfulness=0.69).passes
