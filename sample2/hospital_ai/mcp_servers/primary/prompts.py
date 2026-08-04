"""MCP Prompts exposed by the Primary Clinical Tools Server — doc Table 2.

| Prompt Name                       | Parameters           | Used By                |
|-----------------------------------|----------------------|------------------------|
| discharge-extraction-prompt       | language, doc_types  | Clinical Extractor     |
| ehr-cross-validation-prompt       | patient_id           | Validation Agent       |
| abbreviation-normalization-prompt | source_language      | Normalizer Agent       |
| summary-generation-prompt         | risk_level, audience | Summary Generator      |
| rag-answer-prompt                 | context_length       | Agno RAG Generation    |

Templates live in ``configs/prompts.yaml``. Doc §2.6 is explicit that the RAG
Generation Agent must fetch its prompt via ``get_prompt("rag-answer-prompt")``
rather than using a hardcoded string; the same rule is applied to every agent
here so prompts stay a runtime contract.
"""

from __future__ import annotations

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import ConfigError
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="mcp-prompts")

#: Prompt name -> declared parameters, exactly as Table 2 specifies.
PROMPT_PARAMETERS: dict[str, tuple[str, ...]] = {
    "discharge-extraction-prompt": ("language", "doc_types"),
    "ehr-cross-validation-prompt": ("patient_id",),
    "abbreviation-normalization-prompt": ("source_language",),
    "summary-generation-prompt": ("risk_level", "audience"),
    "rag-answer-prompt": ("context_length",),
}


def render(name: str, **params: object) -> str:
    """Fill a prompt template from ``configs/prompts.yaml``.

    Unsupplied placeholders are left visible as ``{name}`` rather than raising,
    so a partially-parameterised prompt is still usable and the omission is
    obvious in the trace.
    """
    templates = get_settings().prompts
    if name not in templates:
        raise ConfigError(f"Prompt {name!r} is not defined in configs/prompts.yaml")

    template = str(templates[name])
    for key, value in params.items():
        template = template.replace("{" + key + "}", str(value))
    return template.strip()


def register(mcp) -> None:
    """Attach all five prompts of Table 2 to the FastMCP server."""

    @mcp.prompt(
        name="discharge-extraction-prompt",
        description="Instructs the Clinical Extractor Agent how to harvest "
        "structured clinical data from a discharge packet.",
    )
    def discharge_extraction_prompt(language: str = "en", doc_types: str = "discharge_report") -> str:
        return render("discharge-extraction-prompt", language=language, doc_types=doc_types)

    @mcp.prompt(
        name="ehr-cross-validation-prompt",
        description="Instructs the Validation Agent how to compare a discharge "
        "record against the Mock EHR.",
    )
    def ehr_cross_validation_prompt(patient_id: str) -> str:
        return render("ehr-cross-validation-prompt", patient_id=patient_id)

    @mcp.prompt(
        name="abbreviation-normalization-prompt",
        description="Instructs the Normalizer Agent how to translate and expand "
        "medical abbreviations without altering clinical meaning.",
    )
    def abbreviation_normalization_prompt(source_language: str = "en") -> str:
        return render("abbreviation-normalization-prompt", source_language=source_language)

    @mcp.prompt(
        name="summary-generation-prompt",
        description="Instructs the Summary Generator how to write a "
        "patient-friendly discharge summary.",
    )
    def summary_generation_prompt(risk_level: str = "Low", audience: str = "patient") -> str:
        return render("summary-generation-prompt", risk_level=risk_level, audience=audience)

    @mcp.prompt(
        name="rag-answer-prompt",
        description="Grounding instructions for the Agno RAG Generation Agent, "
        "including the mandated out-of-context response.",
    )
    def rag_answer_prompt(context_length: int = 0) -> str:
        return render("rag-answer-prompt", context_length=context_length)

    _log.info("registered MCP prompts", extra={"count": len(PROMPT_PARAMETERS)})
