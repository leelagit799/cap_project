"""EHR Validation Tool — Tool (doc Table 7, §2.4.2).

Cross-checks the extracted discharge data against the Mock EHR REST API and
implements the seven rules of Table 4:

| Rule ID                     | Severity | Action          |
|-----------------------------|----------|-----------------|
| med_omission_check          | Warning  | Flag for review |
| allergy_contradiction_check | Critical | Block discharge |
| diagnosis_mismatch_check    | Warning  | Flag for review |
| follow_up_missing_check     | Critical | Block discharge |
| lab_follow_up_mismatch_check| Warning  | Flag for review |
| discharge_approval_check    | Critical | Block discharge |
| bill_settlement_check       | Critical | Block discharge |

An EHR outage is itself a Critical finding: auto-approving a discharge that
was never verified against the record would be worse than blocking it.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from hospital_ai.core.config import get_settings
from hospital_ai.core.errors import EHRUnavailableError
from hospital_ai.core.logging import get_logger
from hospital_ai.core.retry import retry_async
from hospital_ai.core.schemas import Severity

_log = get_logger(__name__, tool="ehr-validator")


#: Cross-language and brand/generic drug aliases. The dataset ships Spanish,
#: Dutch, German and French discharge notes whose drug names must reconcile
#: against English EHR orders.
DRUG_ALIASES: dict[str, str] = {
    "metformina": "metformin",
    "atorvastatina": "atorvastatin",
    "aspirina": "aspirin",
    "amoxicilline": "amoxicillin",
    "amoxicilina": "amoxicillin",
    "paracetamol": "acetaminophen",
    "paracetamolo": "acetaminophen",
    "salbutamol": "albuterol",
    "furosemida": "furosemide",
    "lisinoprilo": "lisinopril",
    "ciprofloxacine": "ciprofloxacin",
    "azitromicina": "azithromycin",
    "nitrofurantoine": "nitrofurantoin",
    "ondansetrón": "ondansetron",
    "loperamida": "loperamide",
    "hioscina": "hyoscine",
}

#: Drug families that collide with a documented allergy.
ALLERGY_CONFLICTS: dict[str, tuple[str, ...]] = {
    "penicillin": (
        "penicillin", "amoxicillin", "amoxicilline", "amoxicilina",
        "ampicillin", "augmentin", "amoxicillin-clavulanate",
        "amox-clav", "piperacillin", "flucloxacillin", "benzylpenicillin",
    ),
    "sulfa": ("sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim", "sulfadiazine"),
    "latex": (),
    "nsaid": ("ibuprofen", "naproxen", "diclofenac", "ketorolac", "aspirin"),
    "cephalosporin": ("cefazolin", "ceftriaxone", "cephalexin", "cefuroxime"),
}


def canonical_drug(name: str) -> str:
    """Normalise a drug name for comparison across languages and formulations."""
    cleaned = re.sub(r"\s*\([^)]*\)", "", (name or "").strip().lower())
    cleaned = re.sub(r"\b(hfa|mdi|tablet|tablets|caps?|capsules?|inhaler|oral|iv|im)\b", "", cleaned)
    cleaned = re.sub(r"[^a-zà-ÿ0-9\- ]", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return DRUG_ALIASES.get(cleaned, cleaned)


def _finding(
    rule_id: str,
    severity: Severity,
    message: str,
    *,
    field: str | None = None,
    expected: Any = None,
    actual: Any = None,
    weight: int = 0,
    blocking: bool = False,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "severity": severity.value,
        "message": message,
        "field": field,
        "expected": expected,
        "actual": actual,
        "weight": weight,
        "blocking": blocking,
        "resolved": False,
    }


@retry_async(max_attempts=3, base_delay=1.0, max_delay=4.0)
async def fetch_ehr_bundle(patient_id: str) -> dict[str, Any]:
    """Fetch the patient's full EHR record from the Mock EHR REST API."""
    settings = get_settings()
    url = f"http://localhost:{settings.ports.ehr}/patients/{patient_id}/bundle"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        raise EHRUnavailableError(f"Mock EHR unreachable at {url}: {exc}") from exc

    if response.status_code == 404:
        return {}
    if response.status_code >= 500:
        raise EHRUnavailableError(f"Mock EHR returned {response.status_code} for {patient_id}")
    response.raise_for_status()
    return response.json()


# --- Individual Table 4 rules ------------------------------------------------


def check_allergy_contradiction(
    prescriptions: list[dict[str, Any]], allergies: list[str], weight: int
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    prescribed = {canonical_drug(p.get("medicine_name", "")): p for p in prescriptions}

    for allergy in allergies:
        family = allergy.strip().lower()
        conflicts = ALLERGY_CONFLICTS.get(family, (family,))
        for drug, prescription in prescribed.items():
            if not drug:
                continue
            if drug in {canonical_drug(c) for c in conflicts} or family in drug:
                findings.append(
                    _finding(
                        "allergy_contradiction_check",
                        Severity.CRITICAL,
                        f"{prescription.get('medicine_name')} is prescribed despite a "
                        f"documented {allergy} allergy. Discharge blocked.",
                        field="medications",
                        expected=f"no {allergy}-class drug",
                        actual=prescription.get("medicine_name"),
                        weight=weight,
                        blocking=True,
                    )
                )
    return findings


def check_medication_reconciliation(
    prescriptions: list[dict[str, Any]],
    ehr_medications: list[dict[str, Any]],
    weights: dict[str, int],
    high_risk_meds: list[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    discharge = {canonical_drug(p.get("medicine_name", "")) for p in prescriptions} - {""}
    ordered = {canonical_drug(m.get("name", "")) for m in ehr_medications} - {""}
    high_risk = {canonical_drug(m) for m in high_risk_meds}

    for omitted in sorted(ordered - discharge):
        findings.append(
            _finding(
                "med_omission_check",
                Severity.WARNING,
                f"'{omitted}' is on the EHR medication list but absent from the "
                "discharge prescriptions.",
                field="medications",
                expected=omitted,
                actual=None,
                weight=weights.get("medication_omission", 3),
            )
        )

    for added in sorted(discharge - ordered):
        is_high_risk = added in high_risk
        findings.append(
            _finding(
                "high_risk_med_missing_in_ehr" if is_high_risk else "med_omission_check",
                Severity.CRITICAL if is_high_risk else Severity.WARNING,
                f"'{added}' appears on the discharge prescription but has no "
                "corresponding EHR order."
                + (" This is a high-risk medication; pharmacist review required." if is_high_risk else ""),
                field="medications",
                expected=None,
                actual=added,
                weight=weights.get(
                    "high_risk_med_missing_in_ehr" if is_high_risk else "medication_added", 4
                ),
                blocking=is_high_risk,
            )
        )
    return findings


def check_diagnosis_match(
    discharge_icd10: list[str], ehr_primary_dx: list[str], weight: int
) -> list[dict[str, Any]]:
    if not ehr_primary_dx:
        return []
    documented = {code.strip().upper() for code in discharge_icd10 if code}
    expected = {code.strip().upper() for code in ehr_primary_dx}
    if documented & expected:
        return []
    return [
        _finding(
            "diagnosis_mismatch_check",
            Severity.WARNING,
            "Discharge diagnosis does not match the EHR care plan.",
            field="discharge_diagnosis",
            expected=sorted(expected),
            actual=sorted(documented) or None,
            weight=weight,
        )
    ]


def check_follow_up(
    follow_ups: list[str], care_plan: dict[str, Any] | None, weight: int
) -> list[dict[str, Any]]:
    if not care_plan or not care_plan.get("followup_required"):
        return []
    if follow_ups:
        return []
    return [
        _finding(
            "follow_up_missing_check",
            Severity.CRITICAL,
            f"The care plan requires {care_plan.get('speciality')} follow-up within "
            f"{care_plan.get('window_days')} days, but the discharge note documents none. "
            "Discharge blocked.",
            field="follow_up_appointments",
            expected=f"{care_plan.get('speciality')} within {care_plan.get('window_days')} days",
            actual=None,
            weight=weight,
            blocking=True,
        )
    ]


def check_abnormal_labs(
    ehr_labs: list[dict[str, Any]], discharge_text: str, weight: int
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for lab in ehr_labs:
        if not lab.get("abnormal"):
            continue
        documented_in_ehr = bool((lab.get("action_in_ehr") or "").strip())
        mentioned_in_discharge = (lab.get("test", "").lower() in discharge_text.lower())
        if documented_in_ehr or mentioned_in_discharge:
            continue
        findings.append(
            _finding(
                "lab_follow_up_mismatch_check",
                Severity.WARNING,
                f"Abnormal {lab.get('test')} ({lab.get('value')}) has no documented "
                "follow-up action.",
                field="labs",
                expected="documented action",
                actual=lab.get("value"),
                weight=weight,
            )
        )
    return findings


def check_discharge_approval(discharge: dict[str, Any], weight: int) -> list[dict[str, Any]]:
    approved = discharge.get("discharge_approved")
    approved_by = discharge.get("discharge_approved_by") or discharge.get("attending_physician")
    if approved and approved_by:
        return []
    return [
        _finding(
            "discharge_approval_check",
            Severity.CRITICAL,
            "Discharge has not been approved by the treating physician. Discharge blocked.",
            field="discharge_approved",
            expected=True,
            actual=approved,
            weight=weight,
            blocking=True,
        )
    ]


def check_bill_settlement(bill: dict[str, Any] | None, weight: int) -> list[dict[str, Any]]:
    if bill is None:
        return [
            _finding(
                "bill_settlement_check",
                Severity.CRITICAL,
                "No hospital bill was supplied; settlement cannot be confirmed.",
                field="payment_status",
                expected="PAID",
                actual=None,
                weight=weight,
                blocking=True,
            )
        ]

    status = str(bill.get("payment_status") or "UNKNOWN").upper()
    settled = status in {"PAID", "INSURANCE_GUARANTEED"}
    if settled:
        return []
    return [
        _finding(
            "bill_settlement_check",
            Severity.CRITICAL,
            f"Bill status is {status}; settlement or an insurance guarantee letter "
            "is required before release. Discharge blocked.",
            field="payment_status",
            expected="PAID or INSURANCE_GUARANTEED",
            actual=status,
            weight=weight,
            blocking=True,
        )
    ]


# --- Orchestration -----------------------------------------------------------


async def cross_validate(patient_id: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Run all seven Table 4 rules for one patient."""
    settings = get_settings()
    weights = settings.rules["risk_scoring_matrix"]["weights"]
    policies = settings.rules.get("clinical_validation_policies", {})
    high_risk_meds = policies.get("high_risk_meds_need_counseling", [])

    try:
        bundle = await fetch_ehr_bundle(patient_id)
    except EHRUnavailableError as exc:
        _log.error("EHR unavailable", extra={"patient_id": patient_id, "error": str(exc)})
        return {
            "patient_id": patient_id,
            "ehr_available": False,
            "findings": [
                _finding(
                    "ehr_unavailable",
                    Severity.CRITICAL,
                    f"The Mock EHR could not be reached, so this discharge could not "
                    f"be verified against the patient record: {exc}",
                    weight=weights.get("allergy_contradiction", 8),
                    blocking=True,
                )
            ],
        }

    if not bundle:
        return {
            "patient_id": patient_id,
            "ehr_available": True,
            "findings": [
                _finding(
                    "patient_not_in_ehr",
                    Severity.CRITICAL,
                    f"{patient_id} has no record in the EHR; cross-validation is "
                    "impossible. Discharge blocked.",
                    weight=weights.get("allergy_contradiction", 8),
                    blocking=True,
                )
            ],
        }

    discharge = packet.get("discharge_report") or {}
    bill = packet.get("bill")
    prescriptions = discharge.get("medications") or []
    discharge_text = " ".join(
        str(v) for v in (
            discharge.get("discharge_instructions"),
            *(discharge.get("follow_up_appointments") or []),
        ) if v
    )

    findings: list[dict[str, Any]] = []
    findings += check_allergy_contradiction(
        prescriptions, bundle.get("allergies", []), weights.get("allergy_contradiction", 8)
    )
    findings += check_medication_reconciliation(
        prescriptions, bundle.get("medications", []), weights, high_risk_meds
    )
    findings += check_diagnosis_match(
        discharge.get("icd10_codes") or [],
        (bundle.get("patient") or {}).get("primary_dx", []),
        weights.get("diagnosis_mismatch", 4),
    )
    findings += check_follow_up(
        discharge.get("follow_up_appointments") or [],
        bundle.get("care_plan"),
        weights.get("followup_missing", 2),
    )
    findings += check_abnormal_labs(
        bundle.get("labs", []), discharge_text, weights.get("abnormal_lab_unresolved", 3)
    )
    findings += check_discharge_approval(discharge, weights.get("missing_mandatory_field", 3))
    findings += check_bill_settlement(bill, weights.get("bill_unpaid_with_discharge_ok", 5))

    _log.info(
        "cross-validation complete",
        extra={
            "patient_id": patient_id,
            "findings": len(findings),
            "critical": sum(1 for f in findings if f["severity"] == Severity.CRITICAL.value),
        },
    )
    return {
        "patient_id": patient_id,
        "ehr_available": True,
        "ehr_patient": bundle.get("patient"),
        "findings": findings,
        "rules_version": settings.rules_version,
    }


def register(mcp) -> None:
    @mcp.tool(
        name="ehr_validation",
        description=(
            "Cross-check an extracted discharge packet against the Mock EHR "
            "REST API, applying the seven cross-validation rules of Table 4."
        ),
    )
    async def ehr_validation(patient_id: str, packet: dict[str, Any]) -> dict[str, Any]:
        return await cross_validate(patient_id, packet)

    @mcp.tool(
        name="get_ehr_record",
        description="Fetch a patient's full EHR bundle: demographics, "
        "allergies, medication orders, labs and care plan.",
    )
    async def get_ehr_record(patient_id: str) -> dict[str, Any]:
        return await fetch_ehr_bundle(patient_id)
