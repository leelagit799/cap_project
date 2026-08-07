"""P2 — Primary MCP Clinical Tools Server, all six primitives.

These run over a real MCP client/server session with real callbacks, so a
primitive that is only half-wired fails here rather than at demo time.
"""

from __future__ import annotations

import json
from pathlib import Path

import mcp.types as types
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from hospital_ai.core.config import get_settings
from hospital_ai.mcp_servers.primary.server import create_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- Client-side callbacks ---------------------------------------------------


def roots_callback(workspace: Path):
    """The client declares its authorised Root, as the Monitor Agent does."""

    async def _list_roots(context):
        return types.ListRootsResult(
            roots=[types.Root(uri=AnyUrl(workspace.resolve().as_uri()), name="clinical-input")]
        )

    return _list_roots


def sampling_callback(record: dict):
    """Stands in for the agent's LiteLLM client during MCP Sampling."""

    async def _sample(context, params: types.CreateMessageRequestParams):
        record["called"] = True
        record["hints"] = [h.name for h in (params.modelPreferences.hints or [])]
        record["system_prompt"] = params.systemPrompt
        record["text"] = params.messages[0].content.text
        # The client picks the model from the server's hint, exactly as the
        # Normalizer's callback does.
        hint = record["hints"][0] if record["hints"] else "unknown"
        model = "bedrock/amazon.nova-lite-v1:0" if hint == "nova-lite" else "bedrock/cohere.command-r-plus-v1:0"
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Translated: patient discharged in good condition."),
            model=model,
            stopReason="endTurn",
        )

    return _sample


def elicitation_callback(action: str, data: dict | None = None):
    """Stands in for the Streamlit reviewer form."""

    async def _elicit(context, params: types.ElicitRequestParams):
        if action == "accept":
            return types.ElicitResult(action="accept", content=data or {})
        return types.ElicitResult(action=action)

    return _elicit


# --- Roots -------------------------------------------------------------------


class TestRootsPrimitive:
    async def test_watcher_discovers_packets_through_list_roots(self):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            result = await client.call_tool("clinical_watcher", {})
            payload = json.loads(result.content[0].text)

        assert payload["patient_count"] == 6
        # 22 source files, not 18: four patients ship their bill in two
        # encodings (P1020 pdf+json, P1022 png+json, P1023 png+json,
        # P1024 json+txt). OCR sidecars are excluded.
        assert payload["document_count"] == 22
        ids = {p["patient_id"] for p in payload["patients"]}
        assert ids == {"P1019", "P1020", "P1021", "P1022", "P1023", "P1024"}
        assert all(p["complete"] for p in payload["patients"])

    async def test_duplicate_encodings_collapse_to_one_primary_per_type(self):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            result = await client.call_tool("clinical_watcher", {"patient_id": "P1020"})
            payload = json.loads(result.content[0].text)

        patient = payload["patients"][0]
        assert len(patient["documents"]) == 4  # bill ships as both pdf and json
        assert set(patient["primary_documents"]) == {"discharge_report", "lab_report", "bill"}
        # JSON outranks PDF, so the machine-readable bill wins.
        assert patient["primary_documents"]["bill"] == "bills/P1020_bill.json"

    async def test_returned_uris_are_root_relative(self):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            result = await client.call_tool("clinical_watcher", {"patient_id": "P1019"})
            payload = json.loads(result.content[0].text)

        uris = [d["uri"] for d in payload["patients"][0]["documents"]]
        assert uris == [
            "bills/P1019_bill.json",
            "doctor_reports/P1019_thomas_wright.txt",
            "lab_reports/P1019_labs.txt",
        ]
        assert not any(u.startswith("/") for u in uris)

    async def test_ocr_sidecars_are_not_reported_as_source_documents(self):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            result = await client.call_tool("clinical_watcher", {"patient_id": "P1022"})
            payload = json.loads(result.content[0].text)

        documents = payload["patients"][0]["documents"]
        assert not any(d["uri"].endswith(".ocr.txt") for d in documents)

        # The scanned doctor report carries a transcription sidecar; the
        # scanned bill does not, and must not invent one.
        report = next(d for d in documents if d["uri"] == "doctor_reports/P1022_daan_bakker.png")
        assert report["ocr_sidecar"] == "doctor_reports/P1022_daan_bakker.png.ocr.txt"
        bill_png = next(d for d in documents if d["uri"] == "bills/P1022_bill.png")
        assert bill_png["ocr_sidecar"] is None

    async def test_watcher_refuses_to_run_without_declared_roots(self):
        """No roots means no filesystem access — never a config fallback."""
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.call_tool("clinical_watcher", {})

        assert result.isError
        assert "root" in result.content[0].text.lower()

    async def test_path_traversal_is_rejected(self):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            escape = await client.call_tool(
                "clinical_data_harvester", {"uri": "../../etc/passwd"}
            )
            absolute = await client.call_tool(
                "clinical_data_harvester", {"uri": "/etc/passwd"}
            )

        assert escape.isError
        assert absolute.isError
        assert "not accepted" in absolute.content[0].text or "root" in absolute.content[0].text.lower()


# --- Tools -------------------------------------------------------------------


class TestHarvesterTool:
    @pytest.mark.parametrize(
        ("uri", "marker"),
        [
            ("doctor_reports/P1019_thomas_wright.txt", "Thomas Wright"),
            ("doctor_reports/P1021_rohan_gupta.json", "Rohan"),
            ("doctor_reports/P1020_diego_morales.pdf", "Diego"),
            ("doctor_reports/P1022_daan_bakker.png", "Daan"),
        ],
    )
    async def test_extracts_every_format(self, uri, marker):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            result = await client.call_tool("clinical_data_harvester", {"uri": uri})
            payload = json.loads(result.content[0].text)

        assert payload["ok"]
        assert marker in payload["text"]

    async def test_scanned_png_uses_the_ocr_sidecar(self):
        workspace = get_settings().roots.workspace
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, list_roots_callback=roots_callback(workspace)
        ) as client:
            result = await client.call_tool(
                "clinical_data_harvester", {"uri": "doctor_reports/P1023_grace_bennett.png"}
            )
            payload = json.loads(result.content[0].text)

        assert payload["ocr_used"]
        assert payload["ocr_source"].startswith("sidecar:")


# --- Sampling ----------------------------------------------------------------


class TestSamplingPrimitive:
    async def test_dutch_text_routes_to_the_nova_lite_hint(self):
        record: dict = {}
        dutch = (
            "ONTSLAGBRIEF — patiënt Daan Bakker. Diagnose: longontsteking. "
            "Medicatie: Amoxicilline 500 mg driemaal daags. Controle bij de huisarts."
        )
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, sampling_callback=sampling_callback(record)
        ) as client:
            result = await client.call_tool("medical_lang_bridge", {"text": dutch})
            payload = json.loads(result.content[0].text)

        assert record["called"] is True
        assert record["hints"] == ["nova-lite"]
        assert payload["sampling_used"] is True
        assert payload["source_language"] == "nl"
        assert payload["model_used"] == "bedrock/amazon.nova-lite-v1:0"
        assert "Translated:" in payload["translated_text"]

    async def test_hindi_is_detected_by_script(self):
        record: dict = {}
        hindi = "रोगी को अस्पताल से छुट्टी दी गई। दवाएं: मेटफॉर्मिन 500 मिलीग्राम दिन में दो बार।"
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, sampling_callback=sampling_callback(record)
        ) as client:
            result = await client.call_tool("medical_lang_bridge", {"text": hindi})
            payload = json.loads(result.content[0].text)

        assert payload["source_language"] == "hi"
        assert record["hints"] == ["nova-lite"]

    async def test_english_text_skips_sampling_but_expands_abbreviations(self):
        record: dict = {}
        english = "Patient discharged. Diagnosis: T2DM and HTN. Metformin 500 mg BID."
        async with create_connected_server_and_client_session(
            create_server()._mcp_server, sampling_callback=sampling_callback(record)
        ) as client:
            result = await client.call_tool("medical_lang_bridge", {"text": english})
            payload = json.loads(result.content[0].text)

        assert payload["source_language"] == "en"
        assert record.get("called") is None
        assert "Type 2 Diabetes Mellitus" in payload["translated_text"]
        assert "twice daily" in payload["translated_text"]

    async def test_missing_sampling_callback_is_a_typed_error(self):
        """The tool owns no LLM; a client that cannot sample must fail loudly."""
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.call_tool(
                "medical_lang_bridge", {"text": "Le patient est sorti de l'hôpital."}
            )

        assert result.isError
        assert "sampling" in result.content[0].text.lower()


# --- Elicitation -------------------------------------------------------------


PACKET_WITH_SOFT_GAPS = {
    "discharge_report": {
        "patient_id": "P1021", "patient_name": "Rohan Gupta", "age": 58,
        "gender": "Male", "address": None, "admission_date": "2026-05-28",
        "discharge_date": "2026-05-31", "ward": "2A", "bed_no": "07",
        "doctors": ["Dr. Anjali Mehta"], "discharge_diagnosis": ["Type 2 Diabetes Mellitus"],
        "medications": [{
            "sl_no": 1, "medicine_name": "Metformin", "strength": "500 mg",
            "dosage": "1 tablet", "frequency": "BID", "route": "ORAL",
            "period": "30 days", "remarks": "With meals", "total_quantity": "60",
        }],
        "adr_allergy_info": ["Penicillin"], "follow_up_appointments": ["Endocrinology 2026-06-20"],
        "discharge_instructions": "Continue diabetic diet.",
        "discharge_approved_by": "Dr. Anjali Mehta", "discharge_approved": True,
    },
    "lab_report": {
        "patient_id": "P1021", "vendor_name": "Jeevan Rekha", "lab_name": "Pathology",
        "report_date": "2026-05-31", "tests": [{"test": "HbA1c", "value": "6.4"}],
    },
    "bill": {
        "patient_id": "P1021", "hospital_name": "St. Marian", "billing_date": "2026-05-31",
        "line_items": [{"description": "Ward", "total": 900.0}],
        "total_amount": 1200.0, "payment_status": "UNPAID",
    },
}


class TestElicitationPrimitive:
    async def test_accept_fills_the_gap_and_resolves_the_finding(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server,
            elicitation_callback=elicitation_callback(
                "accept", {"address": "14 Nehru Road, Pune 411001"}
            ),
        ) as client:
            result = await client.call_tool(
                "clinical_rules_engine", {"packet": PACKET_WITH_SOFT_GAPS}
            )
            payload = json.loads(result.content[0].text)

        assert payload["elicitation"]["action"] == "accept"
        assert payload["elicitation"]["response"]["address"].startswith("14 Nehru Road")
        address_finding = next(
            f for f in payload["findings"] if f["field"] == "address"
        )
        assert address_finding["resolved"] is True

    async def test_decline_leaves_the_gap_unresolved_for_hitl(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server,
            elicitation_callback=elicitation_callback("decline"),
        ) as client:
            result = await client.call_tool(
                "clinical_rules_engine", {"packet": PACKET_WITH_SOFT_GAPS}
            )
            payload = json.loads(result.content[0].text)

        assert payload["elicitation"]["action"] == "decline"
        address_finding = next(f for f in payload["findings"] if f["field"] == "address")
        assert address_finding["resolved"] is False

    async def test_cancel_escalates(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server,
            elicitation_callback=elicitation_callback("cancel"),
        ) as client:
            result = await client.call_tool(
                "clinical_rules_engine", {"packet": PACKET_WITH_SOFT_GAPS}
            )
            payload = json.loads(result.content[0].text)

        assert payload["elicitation"]["action"] == "cancel"
        assert "escalat" in payload["elicitation"]["note"].lower()

    async def test_blocking_gaps_are_never_elicited(self):
        """A missing patient_id is not a form-fill; it goes straight to HITL."""
        packet = {"discharge_report": {"patient_name": "Unknown"}, "lab_report": None, "bill": None}
        async with create_connected_server_and_client_session(
            create_server()._mcp_server,
            elicitation_callback=elicitation_callback("accept", {"patient_id": "P9999"}),
        ) as client:
            result = await client.call_tool("clinical_rules_engine", {"packet": packet})
            payload = json.loads(result.content[0].text)

        assert payload["discharge_blocked"] is True
        elicited = (payload["elicitation"] or {}).get("fields_requested", [])
        assert "patient_id" not in elicited

    async def test_no_elicitation_when_the_packet_is_complete(self):
        complete = json.loads(json.dumps(PACKET_WITH_SOFT_GAPS))
        complete["discharge_report"]["address"] = "14 Nehru Road, Pune"
        async with create_connected_server_and_client_session(
            create_server()._mcp_server,
            elicitation_callback=elicitation_callback("accept", {}),
        ) as client:
            result = await client.call_tool("clinical_rules_engine", {"packet": complete})
            payload = json.loads(result.content[0].text)

        assert payload["elicitation"] is None
        assert payload["completeness_score"] == 100.0

    async def test_prescription_warning_fields_reduce_completeness_score(self):
        packet = json.loads(json.dumps(PACKET_WITH_SOFT_GAPS))
        packet["discharge_report"]["address"] = "14 Nehru Road, Pune"
        packet["discharge_report"]["medications"][0].pop("remarks")
        packet["discharge_report"]["medications"][0].pop("period")
        async with create_connected_server_and_client_session(
            create_server()._mcp_server,
            elicitation_callback=elicitation_callback("decline"),
        ) as client:
            result = await client.call_tool("clinical_rules_engine", {"packet": packet})
            payload = json.loads(result.content[0].text)

        assert payload["completeness_score"] < 100.0
        assert any(
            f["rule_id"] == "incomplete_prescription_fields" and f["severity"] == "warning"
            for f in payload["findings"]
        )


# --- Resources ---------------------------------------------------------------


class TestResourcesPrimitive:
    async def test_table_1_resources_are_all_exposed(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            static = {str(r.uri) for r in (await client.list_resources()).resources}
            templates = {
                str(t.uriTemplate) for t in (await client.list_resource_templates()).resourceTemplates
            }

        assert static == {
            "resource://clinical-rules/completeness",
            "resource://clinical-rules/cross-validation",
            "resource://report-template/html",
            "resource://medical-abbreviations",
        }
        assert templates == {
            "resource://discharge-report/{patient_id}",
            "resource://lab-report/{patient_id}",
        }

    async def test_completeness_rules_carry_the_version_stamp(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.read_resource(AnyUrl("resource://clinical-rules/completeness"))

        text = result.contents[0].text
        assert "mandatory_clinical_fields" in text
        assert get_settings().rules_version in text

    async def test_abbreviation_dictionary_is_json(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.read_resource(AnyUrl("resource://medical-abbreviations"))

        payload = json.loads(result.contents[0].text)
        assert payload["abbreviation_map"]["BID"] == "twice daily"
        assert payload["icd10_map"]["Hypertension"] == "I10"

    async def test_patient_document_resources_resolve(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            discharge = await client.read_resource(AnyUrl("resource://discharge-report/P1019"))
            labs = await client.read_resource(AnyUrl("resource://lab-report/P1019"))

        assert "Thomas Wright" in discharge.contents[0].text
        assert "P1019" in labs.contents[0].text


# --- Prompts -----------------------------------------------------------------


class TestPromptsPrimitive:
    async def test_table_2_prompts_are_all_exposed(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            names = {p.name for p in (await client.list_prompts()).prompts}

        assert names == {
            "discharge-extraction-prompt",
            "ehr-cross-validation-prompt",
            "abbreviation-normalization-prompt",
            "summary-generation-prompt",
            "rag-answer-prompt",
        }

    async def test_prompt_parameters_are_substituted(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.get_prompt(
                "summary-generation-prompt", {"risk_level": "High", "audience": "caregiver"}
            )

        text = result.messages[0].content.text
        assert "High risk case" in text
        assert "caregiver" in text
        assert "{risk_level}" not in text

    async def test_rag_prompt_carries_the_mandated_refusal_wording(self):
        async with create_connected_server_and_client_session(
            create_server()._mcp_server
        ) as client:
            result = await client.get_prompt("rag-answer-prompt", {"context_length": "1200"})

        assert (
            "I don't know — this information is not available in the patient records."
            in result.messages[0].content.text
        )
