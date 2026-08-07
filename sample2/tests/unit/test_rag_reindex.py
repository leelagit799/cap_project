"""Tests for RAG re-indexing after HITL corrections."""

from __future__ import annotations

import pytest

from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
from hospital_ai.agents.gateway import LocalPromptGateway
from hospital_ai.rag.reindex import reindex_case_record, resolve_record_patient_id
from hospital_ai.rag.store import FaissStore
from hospital_ai.storage import CaseStore
from hospital_ai.ui.service import DashboardService


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path):
    import hospital_ai.rag.store as rag_store_module

    rag_store_module._STORE = FaissStore(tmp_path / "vectors")
    yield
    rag_store_module._STORE = None


def test_resolve_record_patient_id_falls_back_to_case_store(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite")
    store.create_case("CASE-P1024", "P1024", "trace-p1024")
    record = {"discharge_report": {"patient_id": "P1024", "patient_name": "Bram de Vries"}}

    assert resolve_record_patient_id(record, case_id="CASE-P1024", store=store) == "P1024"


def test_save_review_reindexes_updated_medications(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite")
    vector_dir = tmp_path / "vectors"
    store.create_case("CASE-P1024", "P1024", "trace-p1024")
    store.save_record(
        "CASE-P1024",
        {
            "patient_id": "P1024",
            "discharge_report": {
                "patient_id": "P1024",
                "patient_name": "Bram de Vries",
                "medications": [
                    {"medicine_name": "Amoxicilline", "strength": "500 mg"},
                ],
            },
            "bill": {},
        },
    )

    svc = DashboardService(store=store)
    svc.save_review(
        "CASE-P1024",
        reviewer="clinician",
        decision="edit",
        corrections={
            "discharge_report.medications": [
                {"medicine_name": "Azithromycin", "strength": "500 mg"},
                {"medicine_name": "Paracetamol", "strength": "500 mg"},
            ]
        },
    )

    agent = ClinicalRAGAgent(LocalPromptGateway())
    chunks = agent.retrieval.retrieve(
        "What medications was this patient discharged on?",
        top_k=4,
        patient_id="P1024",
    )

    medication_chunks = [chunk for chunk in chunks if chunk.section == "medications"]
    assert medication_chunks
    assert "Azithromycin" in medication_chunks[0].text
    assert "Amoxicilline" not in medication_chunks[0].text


def test_ask_reindexes_before_answering(tmp_path):
    import hospital_ai.rag.store as rag_store_module
    from hospital_ai.agents.agno.rag_agent import ClinicalRAGAgent
    from hospital_ai.agents.gateway import LocalPromptGateway

    rag_store_module._STORE = FaissStore(tmp_path / "vectors")
    store = CaseStore(tmp_path / "cases.sqlite")
    store.create_case("CASE-P1024", "P1024", "trace-p1024")
    store.save_record(
        "CASE-P1024",
        {
            "patient_id": "P1024",
            "discharge_report": {
                "patient_id": "P1024",
                "patient_name": "Bram de Vries",
                "medications": [{"medicine_name": "Azithromycin", "strength": "500 mg"}],
            },
            "bill": {},
        },
    )

    # Seed the vector store with stale medication data for the same patient.
    stale = {
        "patient_id": "P1024",
        "discharge_report": {
            "patient_name": "Bram de Vries",
            "medications": [{"medicine_name": "Amoxicilline", "strength": "500 mg"}],
        },
    }
    reindex_case_record("CASE-P1024-OLD", stale, store=rag_store_module._STORE)

    svc = DashboardService(store=store)
    answer = svc.ask(
        "What medications was this patient discharged on?",
        patient_id="P1024",
        case_id="CASE-P1024",
    )

    assert "Azithromycin" in answer["answer"]
    assert "Amoxicilline" not in answer["answer"]


def test_reindex_case_record_replaces_existing_chunks(tmp_path):
    store = FaissStore(tmp_path / "vectors")
    record = {
        "patient_id": "P1024",
        "discharge_report": {
            "patient_name": "Bram de Vries",
            "medications": [{"medicine_name": "Metformin", "strength": "500 mg"}],
        },
    }
    reindex_case_record("CASE-P1024-A", record, store=store)
    first_size = store.size

    record["discharge_report"]["medications"] = [
        {"medicine_name": "Azithromycin", "strength": "500 mg"}
    ]
    reindex_case_record("CASE-P1024-B", record, store=store)

    assert store.size == first_size
    hits = store.search("medications", top_k=3, patient_id="P1024")
    assert any("Azithromycin" in hit.chunk.text for hit in hits)
    assert not any("Metformin" in hit.chunk.text for hit in hits)
    assert all(hit.chunk.case_id == "CASE-P1024-B" for hit in hits)
