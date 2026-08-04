"""MCP Resources exposed by the Primary Clinical Tools Server — doc Table 1.

| Resource URI                              | Type         | Content                            |
|-------------------------------------------|--------------|------------------------------------|
| resource://clinical-rules/completeness     | TextResource | Completeness rules from rules.yaml |
| resource://clinical-rules/cross-validation | TextResource | Cross-validation rules             |
| resource://discharge-report/{patient_id}   | FileResource | Raw discharge document text        |
| resource://lab-report/{patient_id}         | FileResource | Raw lab report text                |
| resource://report-template/html            | FileResource | HTML discharge summary template    |
| resource://medical-abbreviations           | TextResource | Abbreviation expansion dictionary  |

The Validation Agent reads its rules through these resources rather than off
disk, which is what makes ``rules.yaml`` a runtime-negotiated contract instead
of a local file read.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.documents.loaders import encoding_rank, is_ocr_sidecar, load_document

_log = get_logger(__name__, component="mcp-resources")

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "reporting" / "templates"
HTML_TEMPLATE = TEMPLATE_DIR / "audit_report.html.j2"

#: Which rules.yaml sections belong to which of the two clinical-rules resources.
COMPLETENESS_SECTIONS = (
    "mandatory_clinical_fields",
    "mandatory_prescription_fields",
    "document_requirements",
    "quality_thresholds",
)

CROSS_VALIDATION_SECTIONS = (
    "cross_validation_rules",
    "clinical_validation_policies",
    "risk_scoring_matrix",
    "business_rules",
)


def _subset(sections: tuple[str, ...]) -> str:
    rules = get_settings().rules
    payload = {name: rules[name] for name in sections if name in rules}
    payload["rules_version"] = get_settings().rules_version
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def completeness_rules() -> str:
    """``resource://clinical-rules/completeness``"""
    return _subset(COMPLETENESS_SECTIONS)


def cross_validation_rules() -> str:
    """``resource://clinical-rules/cross-validation``"""
    return _subset(CROSS_VALIDATION_SECTIONS)


def medical_abbreviations() -> str:
    """``resource://medical-abbreviations``"""
    standards = get_settings().rules.get("normalization_standards", {})
    return json.dumps(
        {
            "abbreviation_map": standards.get("abbreviation_map", {}),
            "icd10_map": standards.get("icd10_map", {}),
            "languages_supported": standards.get("language_codes_supported", []),
        },
        indent=2,
        ensure_ascii=False,
    )


def report_template_html() -> str:
    """``resource://report-template/html``"""
    if HTML_TEMPLATE.is_file():
        return HTML_TEMPLATE.read_text(encoding="utf-8")
    return "<!-- audit_report.html.j2 has not been generated yet -->"


def _find_patient_document(patient_id: str, doctype: str) -> Path | None:
    settings = get_settings()
    directory = settings.roots.workspace / settings.roots.doctype_dirs[doctype]
    if not directory.is_dir():
        return None

    candidates = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name.startswith(patient_id) and not is_ocr_sidecar(path)
    ]
    if not candidates:
        return None

    # Prefer a machine-readable form when the dataset ships several encodings
    # of the same document (P1024 has both a .json and a .txt bill).
    return min(candidates, key=encoding_rank)


def _document_text(patient_id: str, doctype: str, label: str) -> str:
    path = _find_patient_document(patient_id, doctype)
    if path is None:
        return f"No {label} on file for {patient_id}."
    content = load_document(path)
    header = f"# {label} — {patient_id} ({path.name})\n\n"
    return header + (content.text or f"[{path.name}: no extractable text]")


def discharge_report_text(patient_id: str) -> str:
    """``resource://discharge-report/{patient_id}``"""
    return _document_text(patient_id, "discharge_report", "Discharge report")


def lab_report_text(patient_id: str) -> str:
    """``resource://lab-report/{patient_id}``"""
    return _document_text(patient_id, "lab_report", "Lab report")


def register(mcp) -> None:
    """Attach all six resources of Table 1 to the FastMCP server."""

    @mcp.resource(
        "resource://clinical-rules/completeness",
        name="clinical-rules-completeness",
        description="Completeness validation rules from configs/rules.yaml.",
        mime_type="text/yaml",
    )
    def _completeness() -> str:
        return completeness_rules()

    @mcp.resource(
        "resource://clinical-rules/cross-validation",
        name="clinical-rules-cross-validation",
        description="Cross-validation rules, risk weights and business rules.",
        mime_type="text/yaml",
    )
    def _cross_validation() -> str:
        return cross_validation_rules()

    @mcp.resource(
        "resource://discharge-report/{patient_id}",
        name="discharge-report",
        description="Raw discharge document text for one patient.",
        mime_type="text/plain",
    )
    def _discharge(patient_id: str) -> str:
        return discharge_report_text(patient_id)

    @mcp.resource(
        "resource://lab-report/{patient_id}",
        name="lab-report",
        description="Raw lab report text for one patient.",
        mime_type="text/plain",
    )
    def _labs(patient_id: str) -> str:
        return lab_report_text(patient_id)

    @mcp.resource(
        "resource://report-template/html",
        name="report-template-html",
        description="Jinja2 template for the clinician-facing HTML audit report.",
        mime_type="text/html",
    )
    def _template() -> str:
        return report_template_html()

    @mcp.resource(
        "resource://medical-abbreviations",
        name="medical-abbreviations",
        description="Medical abbreviation expansions and ICD-10 mappings.",
        mime_type="application/json",
    )
    def _abbreviations() -> str:
        return medical_abbreviations()

    _log.info("registered MCP resources", extra={"count": 6})
