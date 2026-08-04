"""Composite discharge risk scoring.

Implements the ``risk_scoring_matrix`` of ``configs/rules.yaml``: each
unresolved finding contributes its configured weight, the total is banded into
Low / Medium / High, and any hard guardrail forces the High tier regardless of
the arithmetic.

Shared by the Clinical Insight Reporter Tool (Primary MCP) and
``calculate_risk_score`` (Secondary Analytics MCP) so a case cannot be scored
two different ways depending on which server was asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import Recommendation, RiskLevel, Severity

_log = get_logger(__name__, component="risk-engine")

#: Rule ids that map onto entries in ``hitl_hard_guardrails``.
_GUARDRAIL_RULE_IDS = {
    "allergy_contradiction_check": "allergy_contradiction",
    "allergy_contradiction": "allergy_contradiction",
    "high_risk_med_missing_in_ehr": "high_risk_med_missing_in_ehr",
    "incomplete_prescription_fields": "incomplete_prescription_fields",
    "low_translation_confidence": "translation_confidence_below_threshold",
    "translation_confidence_below_threshold": "translation_confidence_below_threshold",
    "rag_unsafe_response": "rag_unsafe_response",
}

#: Service lines that always require human review, per
#: ``clinical_validation_policies``.
_ALWAYS_HITL_SERVICE_LINES = {
    "pediatrics": "service_line_pediatric",
    "paediatrics": "service_line_pediatric",
    "obstetrics": "service_line_obstetric",
    "oncology": "service_line_oncology",
}


@dataclass
class RiskAssessment:
    score: int = 0
    level: RiskLevel = RiskLevel.LOW
    recommendation: Recommendation = Recommendation.APPROVE
    recommendation_text: str = ""
    discharge_blocked: bool = False
    triggered_guardrails: list[str] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)
    sla_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.score,
            "risk_level": self.level.value,
            "recommendation": self.recommendation.value,
            "recommendation_text": self.recommendation_text,
            "discharge_blocked": self.discharge_blocked,
            "triggered_guardrails": self.triggered_guardrails,
            "contributions": self.contributions,
            "sla_seconds": self.sla_seconds,
        }


def _guardrails_for(
    findings: list[dict[str, Any]],
    translation_confidence: float | None,
    service_line: str | None,
    configured: list[str],
) -> list[str]:
    triggered: set[str] = set()

    for finding in findings:
        if finding.get("resolved"):
            continue
        rule_id = str(finding.get("rule_id", ""))
        for prefix, guardrail in _GUARDRAIL_RULE_IDS.items():
            if rule_id == prefix or rule_id.startswith(prefix):
                triggered.add(guardrail)

    if translation_confidence is not None:
        threshold = get_settings().rules["quality_thresholds"]["translation_confidence_min"]
        if translation_confidence < threshold:
            triggered.add("translation_confidence_below_threshold")

    if service_line:
        key = service_line.strip().lower()
        for marker, guardrail in _ALWAYS_HITL_SERVICE_LINES.items():
            if marker in key:
                triggered.add(guardrail)

    return sorted(triggered & set(configured))


def assess(
    findings: list[dict[str, Any]],
    *,
    translation_confidence: float | None = None,
    service_line: str | None = None,
) -> RiskAssessment:
    """Score a case from its unresolved findings."""
    settings = get_settings()
    matrix = settings.rules["risk_scoring_matrix"]
    weights = matrix["weights"]
    thresholds = matrix["thresholds"]
    configured_guardrails = matrix.get("hitl_hard_guardrails", [])
    reporting = settings.rules.get("reporting", {}).get("recommendations", {})
    sla = settings.rules.get("business_rules", {}).get("sla_seconds", {})

    assessment = RiskAssessment()

    for finding in findings:
        if finding.get("resolved"):
            continue
        weight = int(finding.get("weight") or 0)
        if weight == 0:
            # Fall back to the configured weight when a producer omitted one.
            weight = int(weights.get(str(finding.get("rule_id", "")), 0))
        if weight:
            assessment.score += weight
            assessment.contributions.append(
                {
                    "rule_id": finding.get("rule_id"),
                    "severity": finding.get("severity"),
                    "weight": weight,
                }
            )
        if finding.get("blocking"):
            assessment.discharge_blocked = True
        if finding.get("severity") == Severity.CRITICAL.value:
            assessment.discharge_blocked = True

    assessment.triggered_guardrails = _guardrails_for(
        findings, translation_confidence, service_line, configured_guardrails
    )

    if assessment.score <= thresholds["low_max"]:
        assessment.level = RiskLevel.LOW
    elif assessment.score <= thresholds["medium_max"]:
        assessment.level = RiskLevel.MEDIUM
    else:
        assessment.level = RiskLevel.HIGH

    # A hard guardrail escalates regardless of the arithmetic, and doc Table 12
    # forbids auto-approving anything that is blocked.
    if assessment.triggered_guardrails or assessment.discharge_blocked:
        assessment.level = RiskLevel.HIGH
        assessment.discharge_blocked = True

    if assessment.level is RiskLevel.LOW:
        assessment.recommendation = Recommendation.APPROVE
        assessment.recommendation_text = reporting.get("low", "Approve — Auto-release")
        assessment.sla_seconds = sla.get("auto_approve")
    elif assessment.level is RiskLevel.MEDIUM:
        assessment.recommendation = Recommendation.EDIT
        assessment.recommendation_text = reporting.get("medium", "Standard HITL review")
        assessment.sla_seconds = sla.get("hitl_standard")
    else:
        assessment.recommendation = Recommendation.REJECT
        assessment.recommendation_text = reporting.get("high", "Urgent Attention — Block release")
        assessment.sla_seconds = sla.get("hitl_urgent")

    _log.info(
        "risk assessed",
        extra={
            "score": assessment.score,
            "level": assessment.level.value,
            "blocked": assessment.discharge_blocked,
            "guardrails": assessment.triggered_guardrails,
        },
    )
    return assessment
