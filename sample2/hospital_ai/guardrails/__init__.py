"""Responsible AI guardrails — doc Table 12.

| Guardrail          | Module              | Trigger                                  | Action                    |
|--------------------|---------------------|------------------------------------------|---------------------------|
| PII/PHI Redaction  | PIIRedactor         | Patient name, phone, Aadhaar, PAN        | Mask before logging/API   |
| Hallucination      | HallucinationChecker| RAG faithfulness < 0.7                   | Block; regenerate         |
| Prompt Injection   | PromptInjectionGuard| Query matches injection patterns         | Sanitize or reject; alert |
| Toxicity           | ToxicityFilter      | LLM output in clinical instructions      | Filter before including   |
| HITL Escalation    | GuardrailManager    | risk_level=High or discharge_blocked     | Mandatory human review    |

Redaction applies to logs and outbound payloads, never to the clinical record
itself. Masking a patient's name inside the data the validator compares against
the EHR would break reconciliation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="guardrails")


@dataclass
class GuardrailResult:
    guardrail: str
    blocked: bool = False
    detail: str = ""
    sanitized: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return not self.blocked


class PIIRedactor:
    """Masks direct identifiers before text reaches a log or an external API."""

    name = "PIIRedactor"

    PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
        ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
        ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
        (
            "phone",
            re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
            "[PHONE]",
        ),
        ("mrn", re.compile(r"\bMRN[:\s#]*\w+\b", re.IGNORECASE), "[MRN]"),
    )

    def redact(self, text: str, patient_names: list[str] | None = None) -> GuardrailResult:
        if not text:
            return GuardrailResult(self.name, sanitized=text, detail="empty input")

        redacted = text
        found: list[str] = []

        for label, pattern, replacement in self.PATTERNS:
            redacted, count = pattern.subn(replacement, redacted)
            if count:
                found.append(f"{label}x{count}")

        for name in patient_names or []:
            if name and len(name) > 2:
                pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
                redacted, count = pattern.subn("[PATIENT_NAME]", redacted)
                if count:
                    found.append(f"name x{count}")

        return GuardrailResult(
            self.name,
            blocked=False,
            detail=", ".join(found) if found else "no identifiers found",
            sanitized=redacted,
            metadata={"redactions": len(found)},
        )


class PromptInjectionGuard:
    """Rejects attempts to override the assistant's grounding rules."""

    name = "PromptInjectionGuard"

    PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.I),
        re.compile(r"disregard\s+(the\s+)?(system|previous|above)\s+(prompt|instructions|rules)", re.I),
        re.compile(r"forget\s+(everything|all|your\s+instructions)", re.I),
        re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
        re.compile(r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)", re.I),
        re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(developer|admin|root|dan)\b", re.I),
        re.compile(r"\bjailbreak\b", re.I),
        re.compile(r"(bypass|override|disable)\s+(the\s+)?(safety|guardrail|filter|restriction)", re.I),
        re.compile(r"pretend\s+(that\s+)?(you|there)\s+", re.I),
        re.compile(r"</?(system|assistant)>", re.I),
    )

    def check(self, text: str) -> GuardrailResult:
        if not text:
            return GuardrailResult(self.name, detail="empty input")

        matches = [p.pattern for p in self.PATTERNS if p.search(text)]
        if matches:
            _log.warning(
                "prompt injection detected",
                extra={"patterns": len(matches), "excerpt": text[:120]},
            )
            return GuardrailResult(
                self.name,
                blocked=True,
                detail=f"Matched {len(matches)} injection pattern(s).",
                metadata={"patterns": matches},
            )
        return GuardrailResult(self.name, detail="clean")


class HallucinationChecker:
    """Blocks RAG responses whose faithfulness falls below the threshold."""

    name = "HallucinationChecker"
    threshold = 0.7

    def check(self, faithfulness: float) -> GuardrailResult:
        if faithfulness < self.threshold:
            return GuardrailResult(
                self.name,
                blocked=True,
                detail=(
                    f"Faithfulness {faithfulness:.2f} is below {self.threshold}; "
                    "response blocked pending regeneration."
                ),
                metadata={"faithfulness": faithfulness},
            )
        return GuardrailResult(
            self.name, detail=f"faithfulness {faithfulness:.2f}",
            metadata={"faithfulness": faithfulness},
        )


class ToxicityFilter:
    """Screens generated clinical instructions before they reach a patient."""

    name = "ToxicityFilter"

    TERMS: frozenset[str] = frozenset(
        {
            "idiot", "stupid", "moron", "worthless", "hate you", "shut up",
            "kill yourself", "die already", "disgusting", "pathetic",
        }
    )

    #: Advice a discharge summary must never contain on its own authority.
    UNSAFE_ADVICE: tuple[re.Pattern[str], ...] = (
        re.compile(r"stop\s+taking\s+(all\s+)?(your\s+)?(medication|medicine|drugs)", re.I),
        re.compile(r"(double|triple)\s+(the\s+)?dose", re.I),
        re.compile(r"(ignore|skip)\s+(the\s+)?(follow[- ]?up|appointment)", re.I),
        re.compile(r"you\s+(do\s+not|don'?t)\s+need\s+(to\s+see\s+)?(a\s+)?doctor", re.I),
    )

    def check(self, text: str) -> GuardrailResult:
        if not text:
            return GuardrailResult(self.name, detail="empty input")

        lowered = text.lower()
        hits = [term for term in self.TERMS if term in lowered]
        unsafe = [p.pattern for p in self.UNSAFE_ADVICE if p.search(text)]

        if hits or unsafe:
            _log.warning(
                "toxicity filter tripped",
                extra={"toxic_terms": len(hits), "unsafe_advice": len(unsafe)},
            )
            return GuardrailResult(
                self.name,
                blocked=True,
                detail=(
                    f"{len(hits)} toxic term(s) and {len(unsafe)} unsafe clinical "
                    "instruction(s) detected."
                ),
                metadata={"terms": hits, "unsafe_advice": unsafe},
            )
        return GuardrailResult(self.name, detail="clean")


class GuardrailManager:
    """Single entry point, and owner of the mandatory HITL escalation rule."""

    def __init__(self) -> None:
        self.pii = PIIRedactor()
        self.injection = PromptInjectionGuard()
        self.hallucination = HallucinationChecker()
        self.toxicity = ToxicityFilter()
        self.events: list[GuardrailResult] = []

    def _record(self, result: GuardrailResult) -> GuardrailResult:
        self.events.append(result)
        return result

    def redact(self, text: str, patient_names: list[str] | None = None) -> GuardrailResult:
        return self._record(self.pii.redact(text, patient_names))

    def check_prompt_injection(self, text: str) -> GuardrailResult:
        return self._record(self.injection.check(text))

    def check_faithfulness(self, faithfulness: float) -> GuardrailResult:
        return self._record(self.hallucination.check(faithfulness))

    def check_toxicity(self, text: str) -> GuardrailResult:
        return self._record(self.toxicity.check(text))

    def check_hitl_escalation(
        self, risk_level: str, discharge_blocked: bool
    ) -> GuardrailResult:
        """Doc Table 12: High risk or a blocked discharge means no auto-approve."""
        if discharge_blocked or str(risk_level).lower() == "high":
            return self._record(
                GuardrailResult(
                    "GuardrailManager",
                    blocked=True,
                    detail=(
                        f"Mandatory human review: risk_level={risk_level}, "
                        f"discharge_blocked={discharge_blocked}. Automatic release "
                        "is not permitted."
                    ),
                    metadata={"risk_level": risk_level, "discharge_blocked": discharge_blocked},
                )
            )
        return self._record(
            GuardrailResult("GuardrailManager", detail=f"risk_level={risk_level}; auto-release permitted")
        )

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "guardrail": event.guardrail,
                "blocked": event.blocked,
                "detail": event.detail,
                "metadata": event.metadata,
            }
            for event in self.events
        ]


__all__ = [
    "GuardrailManager",
    "GuardrailResult",
    "HallucinationChecker",
    "PIIRedactor",
    "PromptInjectionGuard",
    "ToxicityFilter",
]
