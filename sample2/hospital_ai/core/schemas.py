"""Typed contracts shared by every agent, tool and UI.

Clinical fields are deliberately Optional. Extraction produces a permissive
record and the Validation Agent decides what is missing, using the required /
blocking matrix in doc Table 3. If these models enforced presence, a document
with a missing field would fail to parse instead of producing the
completeness finding the specification calls for.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hospital_ai.core.ids import utc_now_iso


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# --- Enumerations ------------------------------------------------------------


class DocType(str, Enum):
    DISCHARGE_REPORT = "discharge_report"
    LAB_REPORT = "lab_report"
    BILL = "bill"


class Severity(str, Enum):
    """Doc Table 4 severities. Critical findings block discharge."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Recommendation(str, Enum):
    APPROVE = "Approve"
    EDIT = "Edit"
    REJECT = "Reject"


class PaymentStatus(str, Enum):
    PAID = "PAID"
    UNPAID = "UNPAID"
    PARTIAL = "PARTIAL"
    INSURANCE_GUARANTEED = "INSURANCE_GUARANTEED"
    UNKNOWN = "UNKNOWN"


class CaseStatus(str, Enum):
    CREATED = "CREATED"
    EXTRACTED = "EXTRACTED"
    NORMALIZED = "NORMALIZED"
    VALIDATED = "VALIDATED"
    REPORTED = "REPORTED"
    INDEXED = "INDEXED"
    HITL_PENDING = "HITL_PENDING"
    SUMMARY_READY = "SUMMARY_READY"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class ElicitAction(str, Enum):
    """The three MCP Elicitation outcomes the Rules Engine Tool must handle."""

    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"


# --- Source documents --------------------------------------------------------


class SourceDocument(_Base):
    """A file discovered inside the MCP Roots workspace.

    ``uri`` is root-relative. Absolute paths never leave the Watcher Tool.
    """

    patient_id: str
    doc_type: DocType
    uri: str
    media_type: str
    sha256: str
    size_bytes: int
    modified_at: str
    ocr_sidecar: str | None = None


# --- Clinical content --------------------------------------------------------


class Prescription(_Base):
    """One discharge prescription row.

    All nine columns of doc Table 3 / ``rules.yaml``
    ``mandatory_prescription_fields``.
    """

    sl_no: int | None = None
    medicine_name: str | None = None
    strength: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    route: str | None = None
    period: str | None = None
    remarks: str | None = None
    total_quantity: str | None = None


class LabTest(_Base):
    test: str | None = None
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None
    abnormal: bool = False
    documented_action: str | None = None


class BillLineItem(_Base):
    item_code: str | None = None
    description: str | None = None
    qty: float | None = None
    unit_price: float | None = None
    total: float | None = None


class DischargeReport(_Base):
    patient_id: str | None = None
    patient_name: str | None = None
    age: int | None = None
    gender: str | None = None
    address: str | None = None
    admission_date: str | None = None
    discharge_date: str | None = None
    ward: str | None = None
    bed_no: str | None = None
    attending_physician: str | None = None
    consulting_doctors: list[str] = Field(default_factory=list)
    discharge_diagnosis: list[str] = Field(default_factory=list)
    icd10_codes: list[str] = Field(default_factory=list)
    medications: list[Prescription] = Field(default_factory=list)
    adr_allergy_info: list[str] = Field(default_factory=list)
    follow_up_appointments: list[str] = Field(default_factory=list)
    discharge_instructions: str | None = None
    discharge_approved_by: str | None = None
    discharge_approved: bool | None = None
    service_line: str | None = None

    @property
    def doctors(self) -> list[str]:
        """Table 3 names a single ``doctors`` field; the source documents split
        it into attending and consulting."""
        names = [self.attending_physician] if self.attending_physician else []
        return names + list(self.consulting_doctors)


class LabReport(_Base):
    patient_id: str | None = None
    vendor_name: str | None = None
    lab_name: str | None = None
    report_date: str | None = None
    tests: list[LabTest] = Field(default_factory=list)
    comment: str | None = None

    @property
    def abnormal_tests(self) -> list[LabTest]:
        return [t for t in self.tests if t.abnormal]


class Bill(_Base):
    patient_id: str | None = None
    bill_id: str | None = None
    hospital_name: str | None = None
    billing_date: str | None = None
    currency: str | None = None
    line_items: list[BillLineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    payment_status: PaymentStatus = PaymentStatus.UNKNOWN
    payment_method: str | None = None
    notes: str | None = None

    @property
    def is_settled(self) -> bool:
        """Doc Table 4 ``bill_settlement_check``: PAID, or covered by an
        insurance guarantee letter."""
        return self.payment_status in {
            PaymentStatus.PAID,
            PaymentStatus.INSURANCE_GUARANTEED,
        }


class ClinicalRecord(_Base):
    """The Extractor's output and the Normalizer's input/output."""

    patient_id: str
    case_id: str
    discharge_report: DischargeReport | None = None
    lab_report: LabReport | None = None
    bill: Bill | None = None
    source_language: str = "en"
    languages_detected: dict[str, str] = Field(default_factory=dict)
    documents: list[SourceDocument] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    extracted_at: str = Field(default_factory=utc_now_iso)


class NormalizationResult(_Base):
    """Normalizer output — doc §2.3 requires the confidence score."""

    record: ClinicalRecord
    source_language: str
    translation_confidence: float = Field(ge=0.0, le=1.0)
    per_document_confidence: dict[str, float] = Field(default_factory=dict)
    abbreviations_expanded: dict[str, str] = Field(default_factory=dict)
    model_used: str | None = None
    sampling_used: bool = False
    provenance: str = "live"


# --- Validation --------------------------------------------------------------


class Finding(_Base):
    """One validation issue: a missing field (Table 3) or a fired rule (Table 4)."""

    rule_id: str
    severity: Severity
    message: str
    document: DocType | None = None
    field: str | None = None
    expected: Any = None
    actual: Any = None
    weight: int = 0
    blocking: bool = False
    resolved: bool = False
    resolution_note: str | None = None


class ElicitationRecord(_Base):
    """Audit row for one ``ctx.elicit()`` round trip."""

    fields_requested: list[str]
    requested_schema: dict[str, Any] = Field(default_factory=dict)
    action: ElicitAction
    response: dict[str, Any] = Field(default_factory=dict)
    responded_by: str | None = None
    responded_at: str = Field(default_factory=utc_now_iso)


class ValidationResult(_Base):
    case_id: str
    patient_id: str
    run_no: int = 1
    findings: list[Finding] = Field(default_factory=list)
    elicitations: list[ElicitationRecord] = Field(default_factory=list)
    completeness_score: float = 0.0
    risk_score: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    discharge_blocked: bool = False
    recommendation: Recommendation = Recommendation.APPROVE
    recommendation_text: str = ""
    rules_version: str = ""
    translation_confidence: float | None = None
    validated_at: str = Field(default_factory=utc_now_iso)

    @property
    def critical_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.CRITICAL]

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking and not f.resolved]

    @property
    def requires_hitl(self) -> bool:
        """HITL-2 escalation gate.

        Doc Table 12 ``HITL Escalation``: risk_level=High or
        discharge_blocked=True. Blocking fields and Critical rules both set
        ``discharge_blocked``.
        """
        return self.discharge_blocked or self.risk_level is RiskLevel.HIGH


# --- Summary and RAG ---------------------------------------------------------


class SummarySection(_Base):
    """One streamed section — doc Table 10 mandates patient -> meds -> labs ->
    bill -> instructions ordering."""

    name: str
    title: str
    content: str
    order: int


class DischargeSummary(_Base):
    case_id: str
    patient_id: str
    sections: list[SummarySection] = Field(default_factory=list)
    model_used: str | None = None
    generated_at: str = Field(default_factory=utc_now_iso)

    def as_text(self) -> str:
        ordered = sorted(self.sections, key=lambda s: s.order)
        return "\n\n".join(f"## {s.title}\n{s.content}" for s in ordered)


class RagTriad(_Base):
    """Reflection Agent scores — doc Table 5."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0

    @property
    def passes(self) -> bool:
        """Doc Table 12: faithfulness < 0.7 blocks the response."""
        return self.faithfulness >= 0.7


class RetrievedChunk(_Base):
    chunk_id: str
    patient_id: str | None = None
    doc_type: str | None = None
    section: str | None = None
    text: str = ""
    score: float = 0.0


class RagAnswer(_Base):
    question: str
    answer: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    triad: RagTriad = Field(default_factory=RagTriad)
    blocked: bool = False
    block_reason: str | None = None
    prompt_source: str = "mcp:rag-answer-prompt"


#: Doc §2.6 — the exact wording required for out-of-context questions.
OUT_OF_CONTEXT_ANSWER = (
    "I don't know — this information is not available in the patient records."
)


# --- Case tracking -----------------------------------------------------------


class CaseRecord(_Base):
    case_id: str
    patient_id: str
    trace_id: str
    status: CaseStatus = CaseStatus.CREATED
    risk_level: RiskLevel | None = None
    risk_score: int | None = None
    discharge_blocked: bool = False
    rules_version: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
