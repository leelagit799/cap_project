"""AgentCard definitions for all six A2A agents — doc Table 6 and Table 10.

| Agent                       | Framework  | Port | A2A Mode      |
|-----------------------------|------------|------|---------------|
| Clinical Extractor          | LangGraph  | 8100 | non-streaming |
| Clinical Validation         | LangGraph  | 8101 | non-streaming |
| Clinical Normalizer         | LangGraph  | 8102 | non-streaming |
| Discharge Monitor           | Google ADK | 8103 | non-streaming |
| Discharge Summary Generator | Google ADK | 8104 | STREAMING     |
| Clinical RAG Q&A            | Agno       | 8105 | STREAMING     |
"""

from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    APIKeySecurityScheme,
    In,
    SecurityScheme,
)

from hospital_ai.a2a.auth import AUTH_HEADER
from hospital_ai.core.config import get_settings

SECURITY_SCHEME_NAME = "agentAuthToken"

DEFAULT_MODES = ["application/json", "text/plain"]


def _security_schemes() -> dict[str, SecurityScheme]:
    return {
        SECURITY_SCHEME_NAME: SecurityScheme(
            root=APIKeySecurityScheme(
                name=AUTH_HEADER,
                in_=In.header,
                description="Shared secret required on every A2A invocation.",
            )
        )
    }


def build_card(
    *,
    name: str,
    description: str,
    port: int,
    skills: list[AgentSkill],
    streaming: bool,
    push_notifications: bool = True,
    host: str = "localhost",
) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        protocol_version="0.3.0",
        url=f"http://{host}:{port}/",
        preferred_transport="JSONRPC",
        default_input_modes=DEFAULT_MODES,
        default_output_modes=DEFAULT_MODES,
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
            state_transition_history=True,
        ),
        security_schemes=_security_schemes(),
        security=[{SECURITY_SCHEME_NAME: []}],
        skills=skills,
    )


def extractor_card() -> AgentCard:
    ports = get_settings().ports
    return build_card(
        name="Clinical Extractor Agent",
        description=(
            "LangGraph StateGraph agent that extracts structured clinical data "
            "from discharge reports, lab reports and hospital bills in TXT, "
            "JSON, PDF, DOCX and scanned image formats."
        ),
        port=ports.extractor,
        streaming=False,
        skills=[
            AgentSkill(
                id="extract_discharge_packet",
                name="Extract discharge packet",
                description="Harvest a patient's discharge packet into structured clinical JSON.",
                tags=["extraction", "langgraph", "multimodal"],
                examples=["Extract the discharge packet for P1019"],
                input_modes=DEFAULT_MODES,
                output_modes=["application/json"],
            )
        ],
    )


def validator_card() -> AgentCard:
    ports = get_settings().ports
    return build_card(
        name="Clinical Validation Agent",
        description=(
            "LangGraph agent that validates completeness against rules.yaml, "
            "cross-checks the Mock EHR, computes the composite risk score and "
            "elicits missing non-blocking fields from a reviewer."
        ),
        port=ports.validator,
        streaming=False,
        skills=[
            AgentSkill(
                id="validate_discharge",
                name="Validate discharge",
                description=(
                    "Run completeness and cross-validation rules, returning "
                    "findings, risk level and the discharge_blocked decision."
                ),
                tags=["validation", "langgraph", "risk", "elicitation"],
                examples=["Validate the discharge packet for P1022"],
                input_modes=["application/json"],
                output_modes=["application/json"],
            )
        ],
    )


def normalizer_card() -> AgentCard:
    ports = get_settings().ports
    return build_card(
        name="Clinical Normalizer Agent",
        description=(
            "LangGraph agent that detects the source language, translates "
            "clinical content to English via MCP Sampling and expands medical "
            "abbreviations, reporting a translation confidence score."
        ),
        port=ports.normalizer,
        streaming=False,
        skills=[
            AgentSkill(
                id="normalize_clinical_record",
                name="Normalize clinical record",
                description="Translate and normalize a clinical record into standard English.",
                tags=["translation", "langgraph", "sampling", "multilingual"],
                examples=["Normalize the Dutch discharge note for P1024"],
                input_modes=["application/json"],
                output_modes=["application/json"],
            )
        ],
    )


def monitor_card() -> AgentCard:
    ports = get_settings().ports
    return build_card(
        name="Discharge Monitor Agent",
        description=(
            "Google ADK agent that watches the Roots-authorised input "
            "workspace for new discharge packets and starts a case. Discovers "
            "folders through ctx.list_roots(); never receives a raw path."
        ),
        port=ports.monitor,
        streaming=False,
        skills=[
            AgentSkill(
                id="scan_for_discharges",
                name="Scan for discharge packets",
                description="Detect new patient discharge documents inside the declared roots.",
                tags=["monitoring", "adk", "mcp-roots"],
                examples=["Scan the input folder for new discharge packets"],
                input_modes=DEFAULT_MODES,
                output_modes=["application/json"],
            )
        ],
    )


def summary_card() -> AgentCard:
    ports = get_settings().ports
    return build_card(
        name="Discharge Summary Generator",
        description=(
            "Google ADK agent that streams a patient-friendly discharge "
            "summary section by section: patient, medications, labs, bill, "
            "instructions. Refuses to run for a blocked discharge."
        ),
        port=ports.summary,
        streaming=True,
        skills=[
            AgentSkill(
                id="generate_discharge_summary",
                name="Generate discharge summary",
                description="Stream a plain-English discharge summary for an approved case.",
                tags=["summarization", "adk", "streaming"],
                examples=["Generate the discharge summary for P1019"],
                input_modes=["application/json"],
                output_modes=["text/plain", "application/json"],
            )
        ],
    )


def rag_card() -> AgentCard:
    ports = get_settings().ports
    return build_card(
        name="Clinical RAG Q&A Agent",
        description=(
            "Agno agent with five internal roles — indexing, retrieval, "
            "augmentation, generation and reflection — answering questions "
            "grounded in indexed discharge records, scored on the RAG Triad."
        ),
        port=ports.rag,
        streaming=True,
        skills=[
            AgentSkill(
                id="answer_clinical_question",
                name="Answer clinical question",
                description=(
                    "Answer a question from indexed patient records, streaming "
                    "the response. Returns the mandated refusal when the answer "
                    "is not in the records."
                ),
                tags=["rag", "agno", "streaming", "faiss"],
                examples=[
                    "What medications was P1019 discharged on?",
                    "Which patients have an unpaid bill?",
                ],
                input_modes=DEFAULT_MODES,
                output_modes=["text/plain", "application/json"],
            ),
            AgentSkill(
                id="index_case",
                name="Index a case",
                description="Chunk and embed a discharge case into the FAISS vector store.",
                tags=["rag", "agno", "indexing"],
                examples=["Index case CASE-P1019 into the vector store"],
                input_modes=["application/json"],
                output_modes=["application/json"],
            ),
        ],
    )


#: Every A2A agent, keyed by the short name the orchestrator uses.
CARD_BUILDERS = {
    "extractor": extractor_card,
    "validator": validator_card,
    "normalizer": normalizer_card,
    "monitor": monitor_card,
    "summary": summary_card,
    "rag": rag_card,
}


def all_cards() -> dict[str, AgentCard]:
    return {name: builder() for name, builder in CARD_BUILDERS.items()}
