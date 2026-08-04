"""P10 — Host Orchestrator, ADK agents, storage and observability."""

from __future__ import annotations

import contextlib
import json

import mcp.types as types
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from hospital_ai.agents.adk.host import HostOrchestrator
from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
from hospital_ai.agents.gateway import SessionGateway
from hospital_ai.core.config import get_settings
from hospital_ai.core.schemas import CaseStatus
from hospital_ai.mcp_servers.analytics.server import create_server as create_analytics
from hospital_ai.mcp_servers.primary.server import create_server as create_primary
from hospital_ai.observability import Tracer
from hospital_ai.rag.store import FaissStore
from hospital_ai.storage import CaseStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _roots(context):
    workspace = get_settings().roots.workspace.resolve()
    return types.ListRootsResult(
        roots=[types.Root(uri=AnyUrl(workspace.as_uri()), name="clinical-input")]
    )


async def _decline(context, params):
    return types.ElicitResult(action="decline")


async def _echo_sampling(context, params: types.CreateMessageRequestParams):
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=params.messages[0].content.text),
        model="bedrock/amazon.nova-lite-v1:0",
        stopReason="endTurn",
    )


@contextlib.asynccontextmanager
async def orchestrator(tmp_path):
    async with create_connected_server_and_client_session(
        create_primary()._mcp_server,
        list_roots_callback=_roots,
        sampling_callback=_echo_sampling,
        elicitation_callback=_decline,
    ) as primary:
        async with create_connected_server_and_client_session(
            create_analytics()._mcp_server
        ) as analytics:
            gateway = SessionGateway(primary, analytics)
            store = CaseStore(tmp_path / "state.sqlite")
            rag = ClinicalRAGAgent(gateway, store=FaissStore(tmp_path / "vectors"))
            yield HostOrchestrator(gateway, store=store, rag=rag)


class TestWorkflow:
    @pytest.mark.parametrize(
        ("patient_id", "status", "hitl"),
        [
            ("P1019", CaseStatus.SUMMARY_READY, False),
            ("P1020", CaseStatus.SUMMARY_READY, False),
            ("P1021", CaseStatus.HITL_PENDING, True),
            ("P1022", CaseStatus.HITL_PENDING, True),
            ("P1023", CaseStatus.SUMMARY_READY, False),
            ("P1024", CaseStatus.HITL_PENDING, True),
        ],
    )
    async def test_each_case_reaches_its_documented_outcome(
        self, tmp_path, patient_id, status, hitl
    ):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient(patient_id)

        assert outcome.status is status
        assert outcome.requires_hitl is hitl
        assert outcome.errors == []

    async def test_case_ids_and_trace_ids_are_minted_per_case(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            first = await host.process_patient("P1019")
            second = await host.process_patient("P1023")

        assert first.case_id.startswith("CASE-P1019-")
        assert second.case_id.startswith("CASE-P1023-")
        assert first.trace_id != second.trace_id

    async def test_every_case_is_indexed_including_blocked_ones(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            clean = await host.process_patient("P1019")
            blocked = await host.process_patient("P1022")

        assert clean.indexed_chunks > 0
        assert blocked.indexed_chunks > 0
        assert set(host.rag.store.indexed_patients()) == {"P1019", "P1022"}

    async def test_audit_report_artifacts_are_written(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1021")

        from pathlib import Path

        artifacts = outcome.report["artifacts"]
        assert Path(artifacts["json"]).is_file()
        assert Path(artifacts["html"]).is_file()

        payload = json.loads(Path(artifacts["json"]).read_text(encoding="utf-8"))
        assert payload["rules_version"] == get_settings().rules_version
        assert payload["discharge_blocked"] is True
        assert payload["trace_id"] == outcome.trace_id

    async def test_monitor_discovers_all_six_patients(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            discovered = await host.discover()

        assert discovered["patient_count"] == 6
        assert len(discovered["new_patients"]) == 6

    async def test_acknowledged_documents_are_not_reported_as_new(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            first = await host.discover()
            host.monitor.acknowledge(first["patients"])
            second = await host.discover(only_new=True)

        assert second["new_patients"] == []


class TestEscalationGate:
    async def test_blocked_case_never_produces_a_summary(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1022")
            events = [event async for event in host.stream_summary(outcome.case_id)]

        assert len(events) == 1
        assert events[0]["type"] == "blocked"
        assert "blocked pending clinician review" in events[0]["message"]

    async def test_guardrail_manager_records_the_escalation_decision(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1024")

        escalations = [
            event for event in outcome.guardrails if event["guardrail"] == "GuardrailManager"
        ]
        assert escalations
        assert escalations[-1]["blocked"] is True


class TestPersistence:
    async def test_case_and_record_are_stored(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1019")
            case = host.store.get_case(outcome.case_id)
            record = host.store.get_record(outcome.case_id)

        assert case["patient_id"] == "P1019"
        assert case["risk_level"] == "Low"
        assert record["discharge_report"]["patient_name"] == "Thomas Wright"

    async def test_audit_trail_is_append_only_and_ordered(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1019")
            trail = host.store.audit_trail(outcome.case_id)

        actions = [event["action"] for event in trail]
        assert actions[0] == "case_created"
        assert "validation_saved" in actions
        assert "status_changed" in actions

    async def test_revalidation_creates_a_new_run_without_losing_the_first(self, tmp_path):
        """A HITL correction must not erase what the system saw beforehand."""
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1021")
            assert outcome.requires_hitl

            rerun = await host.revalidate(
                outcome.case_id, {"bill.payment_status": "PAID"}
            )
            runs = host.store.validation_runs(outcome.case_id)

        assert rerun.validation.run_no == 2
        assert len(runs) == 2
        assert runs[0]["run_no"] == 1
        # The first run still records the unpaid bill it saw.
        assert any(f["rule_id"] == "bill_settlement_check" for f in runs[0]["findings"])

    async def test_correcting_the_bill_clears_the_settlement_finding(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1021")
            rerun = await host.revalidate(outcome.case_id, {"bill.payment_status": "PAID"})

        assert "bill_settlement_check" not in {f.rule_id for f in rerun.validation.findings}

    async def test_hitl_review_is_recorded(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1022")
            host.store.save_review(
                outcome.case_id,
                reviewer="dr.hart",
                decision="reject",
                notes="Allergy contradiction confirmed; prescription must change.",
            )
            reviews = host.store.get_reviews(outcome.case_id)

        assert reviews[0]["reviewer"] == "dr.hart"
        assert reviews[0]["decision"] == "reject"

    async def test_stats_summarise_the_cohort(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            for patient in ("P1019", "P1022", "P1023"):
                await host.process_patient(patient)
            stats = host.store.stats()

        assert stats["total_cases"] == 3
        assert stats["blocked_cases"] == 1
        assert stats["by_risk_level"]["Low"] == 2


class TestObservability:
    async def test_one_trace_id_covers_the_whole_case(self, tmp_path):
        async with orchestrator(tmp_path) as host:
            outcome = await host.process_patient("P1019")

        sink = get_settings().reports_dir / "traces.jsonl"
        assert sink.is_file()

        records = [
            json.loads(line)
            for line in sink.read_text(encoding="utf-8").splitlines()
            if outcome.trace_id in line
        ]
        assert records
        assert {record["trace_id"] for record in records} == {outcome.trace_id}

        kinds = {record["kind"] for record in records}
        assert "agent" in kinds
        assert "tool" in kinds
        assert "guardrail" in kinds

    def test_spans_capture_duration_and_errors(self):
        tracer = Tracer("trace-unit-test", case_id="CASE-UNIT")

        with tracer.span("ok-span", kind="agent") as span:
            span.output = {"done": True}

        with pytest.raises(ValueError):
            with tracer.span("failing-span", kind="agent"):
                raise ValueError("boom")

        assert tracer.spans[0].duration_ms is not None
        assert tracer.spans[0].error is None
        assert "ValueError: boom" in tracer.spans[1].error

    def test_pii_is_redacted_before_a_span_leaves_the_process(self):
        tracer = Tracer("trace-redaction", case_id="CASE-PII")
        with tracer.span("call", kind="tool", input={"note": "reach me on 555-123-4567"}):
            pass
        assert "[PHONE]" in tracer.spans[0].input["note"]
        assert "555-123-4567" not in json.dumps(tracer.spans[0].input)
