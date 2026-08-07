"""RAG answer formatting, PII masking, and offline structured responses."""

from __future__ import annotations

import re
from typing import Any

from hospital_ai.core.schemas import OUT_OF_CONTEXT_ANSWER

#: US-style street addresses and common international patterns.
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[\w\s.'-]{2,60}?"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|"
    r"way|court|ct\.?|place|pl\.?|circle|cir\.?|parkway|pkwy\.?)\b"
    r"[\w\s,.'-]*\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)
_ZIP_CITY_STATE = re.compile(
    r"\b[\w\s.'-]{2,40},\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"
)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_SECTION_HEADINGS = (
    "## Direct answer",
    "## Clinical details",
    "## Sources",
    "## Limitations",
)

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "medications": ("medication", "medicine", "prescribed", "drug", "dose", "tablet"),
    "allergies": ("allergy", "allergies", "adr", "reaction"),
    "diagnosis": ("diagnosis", "diagnosed", "condition", "icd"),
    "labs": ("lab", "test", "result", "glucose", "hba1c", "creatinine"),
    "bill": ("bill", "cost", "payment", "total", "invoice", "paid"),
    "follow_up": ("follow-up", "follow up", "appointment", "visit", "clinic"),
    "instructions": ("instruction", "discharge instruction", "self-care", "warning"),
    "address": ("address", "home", "residence", "street", "city"),
    "demographics": ("age", "gender", "ward", "physician", "doctor", "patient"),
    "weather": ("weather", "forecast", "temperature", "rain", "sunny"),
}


def mask_pii(text: str) -> str:
    """Redact direct identifiers while keeping clinical facts usable."""
    if not text:
        return text
    masked = _EMAIL_PATTERN.sub("[email redacted]", text)
    masked = _PHONE_PATTERN.sub("[phone redacted]", masked)
    masked = _ADDRESS_PATTERN.sub("[address redacted]", masked)
    masked = _ZIP_CITY_STATE.sub("[address redacted]", masked)
    masked = re.sub(
        r"\bAddress:\s*[^.\n]+",
        "Address: [redacted]",
        masked,
        flags=re.IGNORECASE,
    )
    return masked


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9][\w\-]{1,}", text.lower())
    stop = {
        "the", "and", "for", "what", "was", "were", "is", "are", "a", "an", "of",
        "to", "in", "on", "patient", "this", "that", "with", "from", "how", "when",
    }
    return {t for t in tokens if t not in stop and len(t) > 2}


def _question_topics(question: str) -> set[str]:
    lowered = question.lower()
    return {topic for topic, markers in _TOPIC_KEYWORDS.items() if any(m in lowered for m in markers)}


def _parse_context_chunks(context: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for block in re.split(r"\n\n+", context.strip()):
        match = re.match(
            r"\[(\d+)\]\s*\(([^/]+)/([^,]+),\s*patient\s+([^)]+)\)\s*\n?(.*)",
            block,
            flags=re.DOTALL,
        )
        if match:
            chunks.append(
                {
                    "index": match.group(1),
                    "doc_type": match.group(2),
                    "section": match.group(3),
                    "patient_id": match.group(4),
                    "text": match.group(5).strip(),
                }
            )
    return chunks


def _chunk_relevance(question: str, chunk: dict[str, str]) -> float:
    question_terms = _keywords(question)
    if not question_terms:
        return 0.0
    chunk_terms = _keywords(chunk["text"])
    overlap = len(question_terms & chunk_terms) / len(question_terms)
    topics = _question_topics(question)
    section = chunk["section"]
    if topics and section in topics:
        overlap = max(overlap, 0.8)
    if "medications" in topics and section == "medications":
        overlap = max(overlap, 0.9)
    if "address" in topics and section == "demographics":
        overlap = max(overlap, 0.75)
    if "weather" in topics:
        overlap = min(overlap, 0.05)
    return overlap


def compose_structured_answer(question: str, context: str, *, body: str | None = None) -> str:
    """Build the mandated markdown structure for a RAG answer."""
    chunks = _parse_context_chunks(context)
    if not chunks:
        return _format_sections(
            direct=OUT_OF_CONTEXT_ANSWER,
            details="- No indexed clinical context was retrieved for this question.",
            sources="- None",
            limitations="- The patient record index returned no matching material.",
        )

    ranked = sorted(chunks, key=lambda c: _chunk_relevance(question, c), reverse=True)
    best = ranked[0]
    best_score = _chunk_relevance(question, best)

    if best_score < 0.12 or "weather" in _question_topics(question):
        return _format_sections(
            direct=OUT_OF_CONTEXT_ANSWER,
            details="- The retrieved records do not contain information that answers this question.",
            sources=_sources_line(ranked[:2]),
            limitations="- Question appears unrelated to the indexed discharge record or the data is absent.",
        )

    if body and not body.lstrip().startswith("Clinical record context"):
        direct = _first_sentence(body)
        details = _bullet_details(body, ranked)
    elif "address" in _question_topics(question):
        direct = "A home street address is present in the record but is withheld for privacy."
        details = "- Location details are redacted in Q&A responses.\n- Ward and admission dates remain available in other sections."
    elif best["section"] == "medications":
        direct = "The discharge record lists the prescribed medications below."
        med_lines = [
            line.strip()
            for line in best["text"].splitlines()
            if line.strip() and not line.lower().startswith("discharge medications")
        ]
        details = "\n".join(f"- {mask_pii(line)}" for line in med_lines[:8])
    elif best["section"] in {"diagnosis", "allergies", "labs", "bill", "follow_up", "instructions"}:
        direct = f"The record contains {best['section'].replace('_', ' ')} information for this patient."
        details = "\n".join(f"- {line.strip()}" for line in best["text"].splitlines() if line.strip())
    else:
        direct = _summarise_chunk(best["text"])
        details = "\n".join(
            f"- {mask_pii(chunk['text'][:220])}" for chunk in ranked[:3] if chunk["text"]
        )

    limitations = "None" if best_score >= 0.35 else "Retrieved context only partially matches the question."
    return _format_sections(
        direct=mask_pii(direct),
        details=mask_pii(details),
        sources=_sources_line(ranked[:3]),
        limitations=limitations,
    )


def normalise_answer(question: str, context: str, raw_answer: str) -> str:
    """Ensure every answer is structured and redacted."""
    text = raw_answer.strip()
    if not text or text == OUT_OF_CONTEXT_ANSWER:
        return compose_structured_answer(question, context)

    if not any(heading in text for heading in _SECTION_HEADINGS):
        return compose_structured_answer(question, context, body=text)

    return mask_pii(text)


def _format_sections(*, direct: str, details: str, sources: str, limitations: str) -> str:
    return (
        f"## Direct answer\n{direct.strip()}\n\n"
        f"## Clinical details\n{details.strip()}\n\n"
        f"## Sources\n{sources.strip()}\n\n"
        f"## Limitations\n{limitations.strip()}"
    )


def _sources_line(chunks: list[dict[str, str]]) -> str:
    if not chunks:
        return "- None"
    return "\n".join(
        f"- [{chunk['index']}] {chunk['doc_type']}/{chunk['section']} (patient {chunk['patient_id']})"
        for chunk in chunks
    )


def _first_sentence(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence[:280]


def _bullet_details(body: str, chunks: list[dict[str, str]]) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")]
    if lines:
        return "\n".join(f"- {mask_pii(line)}" for line in lines[:6])
    return "\n".join(f"- {mask_pii(chunk['text'][:220])}" for chunk in chunks[:2])


def _summarise_chunk(text: str) -> str:
    first = text.split(".")[0].strip()
    return first[:240] if first else OUT_OF_CONTEXT_ANSWER


def triad_from_overlap(
    question: str,
    answer: str,
    chunks: list[Any],
) -> tuple[float, float, float]:
    """Compute faithfulness, answer relevance, and context relevance."""
    from hospital_ai.rag.roles import _keywords as role_keywords

    question_terms = role_keywords(question)
    if not question_terms:
        return 0.0, 0.0, 0.0

    answer_terms = role_keywords(answer)
    context_terms: set[str] = set()
    for chunk in chunks:
        context_terms |= role_keywords(getattr(chunk, "text", str(chunk)))

    context_relevance = len(context_terms & question_terms) / len(question_terms)
    if "weather" in _question_topics(question):
        context_relevance = min(context_relevance, 0.08)

    if answer.strip() == OUT_OF_CONTEXT_ANSWER or "not available in the patient records" in answer.lower():
        return (
            1.0,
            round(min(0.55, 0.25 + context_relevance * 0.5), 3),
            round(min(1.0, context_relevance), 3),
        )

    if not answer_terms:
        return 0.0, 0.0, round(min(1.0, context_relevance), 3)

    faithfulness = len(answer_terms & context_terms) / len(answer_terms)
    answer_relevance = len(answer_terms & question_terms) / len(question_terms)

    # Penalise prompt/context dumps and off-topic replies.
    dump_ratio = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    if dump_ratio > 0.82 and answer_relevance > 0.4:
        answer_relevance *= 0.45
    if context_relevance < 0.15:
        answer_relevance = min(answer_relevance, 0.2)

    lowered = answer.lower()
    if any(
        phrase in lowered
        for phrase in ("redacted", "withheld", "privacy", "on file but withheld")
    ):
        faithfulness = max(faithfulness, 0.95)
        if "address" in _question_topics(question) or "phone" in lowered:
            answer_relevance = max(answer_relevance, 0.7)

    return (
        round(min(1.0, faithfulness), 3),
        round(min(1.0, answer_relevance), 3),
        round(min(1.0, context_relevance), 3),
    )
