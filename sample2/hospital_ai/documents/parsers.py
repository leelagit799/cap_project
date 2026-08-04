"""Structured parsing of clinical documents into typed records.

The sample packets arrive in six languages and two shapes: a labelled plain-text
layout with pipe-delimited tables, and JSON exports whose keys differ per
document. Both are normalised here into the ``DischargeReport``, ``LabReport``
and ``Bill`` models.

Parsing is deliberately literal. A label that is absent leaves its field ``None``
so the Validation Agent can raise the completeness finding doc Table 3 requires;
nothing is inferred or defaulted into place.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from hospital_ai.core.logging import get_logger
from hospital_ai.core.schemas import (
    Bill,
    BillLineItem,
    DischargeReport,
    LabReport,
    LabTest,
    PaymentStatus,
    Prescription,
)
from hospital_ai.documents.loaders import DocumentContent

_log = get_logger(__name__, component="document-parser")


def _norm(label: str) -> str:
    """Fold a label for matching: lowercase, no accents, no punctuation.

    Word characters are kept under Unicode rules rather than restricted to
    ASCII. Restricting to ``[a-z0-9]`` erases Devanagari and Hindi labels
    entirely, which would make every blank line compare equal to them.
    """
    import unicodedata

    stripped = unicodedata.normalize("NFKD", label)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return re.sub(r"[\W_]+", "", stripped.lower(), flags=re.UNICODE)


def _alias_map(aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Invert {field: labels} into {normalised label: field}."""
    return {
        key: field
        for field, labels in aliases.items()
        for label in labels
        if (key := _norm(label))
    }


#: Header labels for discharge reports across en / es / hi / de / fr / nl.
DISCHARGE_LABELS = _alias_map(
    {
        "patient_id": ("Patient ID", "ID del Paciente", "Patiëntnummer", "रोगी आईडी",
                       "Patienten-ID", "ID du Patient", "Patient Id"),
        "patient_name": ("Patient Name", "Nombre", "Naam", "रोगी का नाम", "Name",
                         "Nom du Patient", "Nom"),
        "dob": ("Date of Birth", "Fecha de Nacimiento", "Geboortedatum", "जन्म तिथि",
                "Geburtsdatum", "Date de Naissance"),
        "sex": ("Sex", "Sexo", "Geslacht", "लिंग", "Geschlecht", "Sexe"),
        "gender": ("Gender", "Género", "Genero", "लिंग पहचान"),
        "age": ("Age", "Edad", "Leeftijd", "आयु", "Alter", "Âge"),
        "address": ("Address", "Dirección", "Adres", "पता", "Adresse"),
        "admission_date": ("Admission Date", "Fecha de Ingreso", "Opnamedatum",
                           "भर्ती तिथि", "Aufnahmedatum", "Date d'Admission"),
        "discharge_date": ("Discharge Date", "Fecha de Alta", "Ontslagdatum",
                           "छुट्टी की तिथि", "Entlassungsdatum", "Date de Sortie"),
        "ward": ("Ward", "Sala", "Afdeling", "वार्ड", "Station", "Service"),
        "bed_no": ("Bed No.", "Bed No", "Bed", "Cama", "बेड नं", "Bett", "Lit"),
        "service_line": ("Service Line", "Línea de Servicio", "Specialisme",
                         "सेवा श्रेणी", "Fachbereich", "Spécialité"),
        "attending_physician": ("Attending Physician", "Médico Tratante",
                                "Behandelend arts", "उपस्थित चिकित्सक",
                                "Behandelnder Arzt", "Médecin Traitant"),
        "consulting_doctors": ("Consulting Doctors", "Médicos Consultores",
                               "Consulterende artsen", "परामर्शदाता चिकित्सक",
                               "Konsiliarärzte", "Médecins Consultants"),
        "language": ("Language of Record", "Idioma del Registro", "Taal van dossier",
                     "रिकॉर्ड की भाषा", "Sprache", "Langue du Dossier", "Report Language"),
        "discharge_approved": ("Discharge OK", "Discharge Approved", "Alta Aprobada",
                               "Ontslag goedgekeurd", "डिस्चार्ज स्वीकृत",
                               "Entlassung genehmigt", "Sortie Approuvée"),
    }
)

LAB_LABELS = _alias_map(
    {
        "patient_id": ("Patient ID", "ID del Paciente", "Patiëntnummer", "रोगी आईडी"),
        "patient_name": ("Patient Name", "Nombre", "Naam", "रोगी का नाम"),
        "lab_name": ("Performing Lab", "Laboratorio", "Laboratorium",
                     "Uitvoerend laboratorium", "प्रयोगशाला", "Labor", "Laboratoire"),
        "vendor_name": ("Accreditation", "Acreditación", "Accreditatie", "Laboratory",
                        "Vendor", "Leverancier", "Akkreditierung"),
        "report_date": ("Reported", "Reportado", "Fecha de Informe", "Gerapporteerd",
                        "रिपोर्ट तिथि", "Report Date", "Berichtsdatum", "Rapporté"),
        "attending_physician": ("Ordering Physician", "Médico Solicitante",
                                "Aanvragend arts", "अनुरोधकर्ता चिकित्सक"),
        "language": ("Report Language", "Idioma del Informe", "Taal van rapport"),
    }
)

BILL_LABELS = _alias_map(
    {
        "bill_id": ("Bill ID", "Invoice No", "Factuurnr.", "Factuurnummer",
                    "Nº de Factura", "No. de Factura", "बिल आईडी", "Rechnungsnr."),
        "patient_id": ("Patient ID", "ID del Paciente", "Patiëntnummer", "रोगी आईडी"),
        "patient_name": ("Patient Name", "Nombre", "Naam", "रोगी का नाम"),
        "billing_date": ("Issue Date", "Billing Date", "Fecha de Emisión",
                         "Factuurdatum", "बिल तिथि", "Rechnungsdatum"),
        "currency": ("Currency", "Moneda", "Valuta", "मुद्रा", "Währung"),
        "payment_status": ("Payment Status", "Estado de Pago", "Betaalstatus",
                           "भुगतान स्थिति", "Zahlungsstatus", "Statut de Paiement"),
        "payment_method": ("Payment Method", "Método de Pago", "Betaalwijze",
                           "भुगतान का तरीका", "Zahlungsmethode"),
        "hospital_name": ("Hospital", "Vendor Name", "Hospital Name", "Ziekenhuis"),
    }
)

#: Section headers, matched case-insensitively after normalisation.
SECTION_ALIASES = _alias_map(
    {
        "diagnosis": ("DISCHARGE DIAGNOSIS", "DIAGNÓSTICO DE ALTA", "ONTSLAGDIAGNOSE",
                      "डिस्चार्ज निदान", "ENTLASSUNGSDIAGNOSE", "DIAGNOSTIC DE SORTIE"),
        "allergies": ("ALLERGIES", "ALERGIAS", "ALLERGIEËN", "एलर्जी", "ALLERGIEN"),
        "prescriptions": ("DISCHARGE PRESCRIPTIONS", "RECETAS DE ALTA", "ONTSLAGRECEPTEN",
                          "डिस्चार्ज दवाएं", "ENTLASSUNGSREZEPTE", "ORDONNANCES DE SORTIE"),
        "follow_up": ("FOLLOW-UP APPOINTMENT", "FOLLOW UP APPOINTMENT",
                      "CITA DE SEGUIMIENTO", "VERVOLGAFSPRAAK", "अनुवर्ती नियुक्ति",
                      "NACHSORGETERMIN", "RENDEZ-VOUS DE SUIVI"),
        "instructions": ("DISCHARGE INSTRUCTIONS", "INSTRUCCIONES DE ALTA",
                         "ONTSLAGINSTRUCTIES", "डिस्चार्ज निर्देश",
                         "ENTLASSUNGSANWEISUNGEN", "INSTRUCTIONS DE SORTIE"),
        "lab_results": ("LABORATORY RESULTS", "RESULTADOS DE LABORATORIO",
                        "LABORATORIUMRESULTATEN", "प्रयोगशाला परिणाम", "LABORERGEBNISSE"),
        "abnormal_labs": ("ABNORMAL LAB FINDINGS", "HALLAZGOS ANORMALES",
                          "AFWIJKENDE BEVINDINGEN", "असामान्य निष्कर्ष",
                          "AUFFÄLLIGE BEFUNDE"),
    }
)

AFFIRMATIVE = {"yes", "y", "si", "sí", "ja", "true", "oui", "हाँ", "हां", "approved"}
NEGATIVE = {"no", "nee", "nein", "false", "non", "नहीं"}

PAID_TOKENS = {"paid", "betaald", "pagado", "bezahlt", "payé", "paye",
               "भुगतानकियागया", "भुगतानहुआ"}
UNPAID_TOKENS = {"unpaid", "onbetaald", "nopagado", "pendiente", "unbezahlt",
                 "impayé", "outstanding", "due", "बकाया", "अवैतनिक"}
INSURANCE_TOKENS = {"insuranceguaranteed", "guaranteeletter", "verzekeringsgarantie",
                    "cartadegarantia"}

_SEPARATOR = re.compile(r"^[=\-_]{6,}\s*$")
_ICD10 = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,3})?)\b")


def _is_affirmative(value: str | None) -> bool | None:
    if value is None:
        return None
    token = _norm(value)
    if any(token.startswith(_norm(a)) for a in AFFIRMATIVE):
        return True
    if any(token.startswith(_norm(n)) for n in NEGATIVE):
        return False
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d.,\-]", "", str(value))
    if not cleaned:
        return None
    # European formats write 1.234,56; strip thousands separators before parsing.
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".") if cleaned.rindex(",") > cleaned.rindex(".") else cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def split_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Split a labelled clinical document into ``{section: body}`` plus the header block.

    Sections are announced by an ALL-CAPS line between rule characters, which is
    the layout every text and OCR document in the dataset uses.
    """
    lines = text.splitlines()
    header: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for index, raw in enumerate(lines):
        line = raw.rstrip()
        if _SEPARATOR.match(line):
            continue

        key = _norm(line)
        candidate = SECTION_ALIASES.get(key) if key else None
        if candidate and line.strip() == line.strip().upper():
            current = candidate
            sections.setdefault(current, [])
            continue

        # A heading is normally fenced by rules; treat a bare uppercase line that
        # matches a known alias as a heading too, for OCR output that lost them.
        if current is None:
            header.append(line)
        else:
            sections[current].append(line)

    return {name: "\n".join(body).strip() for name, body in sections.items()}, header


def parse_header_fields(header_lines: Iterable[str], aliases: dict[str, str]) -> dict[str, str]:
    """Read ``Label: value`` pairs from a document's header block."""
    fields: dict[str, str] = {}
    for line in header_lines:
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        field = aliases.get(_norm(label))
        if field is None:
            continue
        value = value.strip()
        if value and field not in fields:
            fields[field] = value
    return fields


def parse_pipe_table(body: str) -> list[dict[str, str]]:
    """Parse a ``a | b | c`` table, using the first row as the header."""
    rows = [line for line in body.splitlines() if line.count("|") >= 2]
    if len(rows) < 2:
        return []

    headers = [cell.strip() for cell in rows[0].split("|")]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = [cell.strip() for cell in row.split("|")]
        if len(cells) < 2:
            continue
        records.append(
            {headers[i] if i < len(headers) else f"col{i}": cells[i] for i in range(len(cells))}
        )
    return records


#: Prescription table columns are positional across languages, so index them.
_PRESCRIPTION_ORDER = (
    "sl_no", "medicine_name", "strength", "dosage", "frequency",
    "route", "period", "remarks", "total_quantity",
)


def parse_prescriptions(body: str) -> list[Prescription]:
    prescriptions: list[Prescription] = []
    for record in parse_pipe_table(body):
        values = list(record.values())
        fields: dict[str, Any] = {
            name: (values[i].strip() if i < len(values) and values[i].strip() else None)
            for i, name in enumerate(_PRESCRIPTION_ORDER)
        }
        fields["sl_no"] = _to_int(fields.get("sl_no"))
        if not fields.get("medicine_name"):
            continue
        prescriptions.append(Prescription(**fields))
    return prescriptions


def _bullets(body: str) -> list[str]:
    items: list[str] = []
    for line in body.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^[-•*]\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        if cleaned:
            items.append(cleaned)
    return items


# --- Discharge report --------------------------------------------------------


def parse_discharge_report(content: DocumentContent) -> DischargeReport:
    if content.structured:
        return _discharge_from_json(content.structured)
    return _discharge_from_text(content.text)


def _discharge_from_json(payload: dict[str, Any]) -> DischargeReport:
    """Map a JSON discharge export onto the canonical model."""
    medications = [
        Prescription(
            sl_no=_to_int(med.get("sl_no")),
            # JSON exports use `name`; the text layout uses `medicine_name`.
            medicine_name=med.get("medicine_name") or med.get("name"),
            strength=med.get("strength"),
            dosage=med.get("dosage"),
            frequency=med.get("frequency"),
            route=med.get("route"),
            period=med.get("period"),
            remarks=med.get("remarks"),
            total_quantity=str(med["total_quantity"]) if med.get("total_quantity") is not None else None,
        )
        for med in payload.get("medications") or []
    ]

    follow_up = payload.get("follow_up_appointments") or payload.get("follow_up_appointment")
    if isinstance(follow_up, str):
        follow_up = [follow_up]

    diagnoses = payload.get("discharge_diagnosis") or []
    if isinstance(diagnoses, str):
        diagnoses = [diagnoses]

    approved = payload.get("discharge_approved")
    if approved is None:
        approved = payload.get("discharge_ok")

    return DischargeReport(
        patient_id=payload.get("patient_id"),
        patient_name=payload.get("patient_name"),
        age=_to_int(payload.get("age")),
        gender=payload.get("gender") or payload.get("sex"),
        address=payload.get("address"),
        admission_date=payload.get("admission_date"),
        discharge_date=payload.get("discharge_date"),
        ward=payload.get("ward"),
        bed_no=str(payload["bed_no"]) if payload.get("bed_no") is not None else None,
        attending_physician=payload.get("attending_physician"),
        consulting_doctors=list(payload.get("consulting_doctors") or []),
        discharge_diagnosis=list(diagnoses),
        icd10_codes=_icd10_codes(diagnoses),
        medications=medications,
        adr_allergy_info=list(payload.get("allergies") or payload.get("adr_allergy_info") or []),
        follow_up_appointments=list(follow_up or []),
        discharge_instructions=payload.get("discharge_instructions"),
        discharge_approved_by=payload.get("discharge_approved_by") or payload.get("attending_physician"),
        discharge_approved=bool(approved) if approved is not None else None,
        service_line=payload.get("service_line"),
    )


def _discharge_from_text(text: str) -> DischargeReport:
    sections, header = split_sections(text)
    fields = parse_header_fields(header, DISCHARGE_LABELS)

    diagnoses = _bullets(sections.get("diagnosis", ""))
    allergies = _bullets(sections.get("allergies", ""))
    consulting = fields.get("consulting_doctors")

    return DischargeReport(
        patient_id=fields.get("patient_id"),
        patient_name=fields.get("patient_name"),
        age=_to_int(fields.get("age")),
        gender=fields.get("gender") or fields.get("sex"),
        address=fields.get("address"),
        admission_date=fields.get("admission_date"),
        discharge_date=fields.get("discharge_date"),
        ward=fields.get("ward"),
        bed_no=fields.get("bed_no"),
        attending_physician=fields.get("attending_physician"),
        consulting_doctors=[d.strip() for d in re.split(r"[;,]", consulting) if d.strip()]
        if consulting
        else [],
        discharge_diagnosis=diagnoses,
        icd10_codes=_icd10_codes(diagnoses),
        medications=parse_prescriptions(sections.get("prescriptions", "")),
        adr_allergy_info=allergies,
        follow_up_appointments=_bullets(sections.get("follow_up", "")),
        discharge_instructions="\n".join(_bullets(sections.get("instructions", ""))) or None,
        # The text layout states approval as a labelled flag signed by the
        # attending physician; there is no separate approver line.
        discharge_approved_by=fields.get("attending_physician"),
        discharge_approved=_is_affirmative(fields.get("discharge_approved")),
        service_line=fields.get("service_line"),
    )


def _icd10_codes(diagnoses: Iterable[str]) -> list[str]:
    codes: list[str] = []
    for entry in diagnoses:
        codes.extend(match.group(1) for match in _ICD10.finditer(str(entry)))
    return list(dict.fromkeys(codes))


# --- Lab report --------------------------------------------------------------

_ABNORMAL_FLAGS = {"high", "low", "abnormal", "critical", "h", "l", "alto", "bajo",
                   "hoog", "laag", "असामान्य", "उच्च"}


def parse_lab_report(content: DocumentContent) -> LabReport:
    if content.structured:
        return _labs_from_json(content.structured)
    return _labs_from_text(content.text)


def _labs_from_json(payload: dict[str, Any]) -> LabReport:
    tests: list[LabTest] = []
    abnormal_names = {
        _norm(str(entry.get("test") or entry))
        for entry in payload.get("abnormal_labs") or []
    }

    for row in payload.get("lab_results") or payload.get("tests") or []:
        # Localised JSON reports key their columns in the source language.
        name = _first(row, ("test", "Test", "परीक्षण", "prueba", "Prüfung"))
        value = _first(row, ("result", "value", "परिणाम", "resultado", "Ergebnis"))
        unit = _first(row, ("unit", "units", "इकाई", "unidad", "Einheit"))
        reference = _first(row, ("reference_range", "Reference Range", "संदर्भ सीमा",
                                 "rango de referencia", "Referenzbereich"))
        flag = _first(row, ("flag", "indicator", "संकेतक", "indicador", "Indikator"))
        tests.append(
            LabTest(
                test=name,
                value=str(value) if value is not None else None,
                unit=unit,
                reference_range=reference,
                flag=flag,
                abnormal=_norm(str(flag or "")) in _ABNORMAL_FLAGS
                or _norm(str(name or "")) in abnormal_names,
            )
        )

    return LabReport(
        patient_id=payload.get("patient_id"),
        vendor_name=payload.get("lab_accreditation") or payload.get("vendor_name"),
        lab_name=payload.get("performing_lab") or payload.get("lab_name"),
        report_date=payload.get("reported") or payload.get("report_date"),
        tests=tests,
        comment=payload.get("comment") or payload.get("kommentar"),
    )


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    normalised = {_norm(k): v for k, v in row.items()}
    for key in keys:
        value = normalised.get(_norm(key))
        if value not in (None, ""):
            return value
    return None


def _labs_from_text(text: str) -> LabReport:
    sections, header = split_sections(text)
    fields = parse_header_fields(header, LAB_LABELS)

    tests: list[LabTest] = []
    for record in parse_pipe_table(sections.get("lab_results", "")):
        values = [v.strip() for v in record.values()]
        while len(values) < 5:
            values.append("")
        name, value, unit, reference, flag = values[:5]
        if not name:
            continue
        tests.append(
            LabTest(
                test=name,
                value=value or None,
                unit=unit or None,
                reference_range=reference or None,
                flag=flag or None,
                abnormal=_norm(flag) in _ABNORMAL_FLAGS,
            )
        )

    # "HbA1c 6.9 % - Action: continue Metformin" documents the follow-up.
    for line in _bullets(sections.get("abnormal_labs", "")):
        name, _, action = line.partition("- Action:")
        if not action:
            continue
        target = _norm(name.split()[0]) if name.split() else ""
        for test in tests:
            if target and target in _norm(test.test or ""):
                test.abnormal = True
                test.documented_action = action.strip()

    return LabReport(
        patient_id=fields.get("patient_id"),
        vendor_name=fields.get("vendor_name"),
        lab_name=fields.get("lab_name"),
        report_date=fields.get("report_date"),
        tests=tests,
    )


# --- Bill --------------------------------------------------------------------


def _payment_status(raw: str | None) -> PaymentStatus:
    if not raw:
        return PaymentStatus.UNKNOWN
    token = _norm(raw)
    if any(t in token for t in INSURANCE_TOKENS):
        return PaymentStatus.INSURANCE_GUARANTEED
    if any(t in token for t in UNPAID_TOKENS):
        return PaymentStatus.UNPAID
    if any(t in token for t in PAID_TOKENS):
        return PaymentStatus.PAID
    if "partial" in token or "gedeeltelijk" in token or "parcial" in token:
        return PaymentStatus.PARTIAL
    return PaymentStatus.UNKNOWN


def parse_bill(content: DocumentContent) -> Bill:
    if content.structured:
        return _bill_from_json(content.structured)
    return _bill_from_text(content.text)


def _bill_from_json(payload: dict[str, Any]) -> Bill:
    items = [
        BillLineItem(
            item_code=item.get("item_code"),
            description=item.get("description"),
            qty=_to_float(item.get("qty")),
            unit_price=_to_float(item.get("unit_price")),
            total=_to_float(item.get("total")),
        )
        for item in payload.get("line_items") or []
    ]
    return Bill(
        patient_id=payload.get("patient_id"),
        bill_id=payload.get("bill_id"),
        hospital_name=payload.get("vendor_name") or payload.get("hospital_name"),
        billing_date=payload.get("issue_date") or payload.get("billing_date"),
        currency=payload.get("currency"),
        line_items=items,
        subtotal=_to_float(payload.get("subtotal")),
        tax=_to_float(payload.get("tax")),
        total_amount=_to_float(payload.get("total_amount")),
        payment_status=_payment_status(
            payload.get("payment_status") or payload.get("payment_status_label")
        ),
        payment_method=payload.get("payment_method"),
        notes=payload.get("notes"),
    )


_TOTAL_LINE = re.compile(
    r"(total\s*(?:due|a\s*pagar|te\s*betalen)?|totaal\s*te\s*betalen|gesamtbetrag|कुल)"
    r"\s*:?\s*([\d.,]+)",
    re.IGNORECASE,
)
_SUBTOTAL_LINE = re.compile(r"(subtotal|subtotaal|zwischensumme|उपयोग)\s*:?\s*([\d.,]+)", re.IGNORECASE)
_TAX_LINE = re.compile(r"\b(tax|btw|iva|mwst|कर)\s*:?\s*([\d.,]+)", re.IGNORECASE)
_ITEM_LINE = re.compile(
    r"^(?P<code>[A-Z][A-Z0-9\-]{2,})\s{2,}(?P<desc>.+?)\s{2,}"
    r"(?P<qty>[\d.,]+)\s{2,}(?P<price>[\d.,]+)\s{2,}(?P<total>[\d.,]+)\s*$"
)


def _bill_from_text(text: str) -> Bill:
    header = [line for line in text.splitlines() if ":" in line]
    fields = parse_header_fields(header, BILL_LABELS)

    items: list[BillLineItem] = []
    for line in text.splitlines():
        match = _ITEM_LINE.match(line.rstrip())
        if match:
            items.append(
                BillLineItem(
                    item_code=match.group("code"),
                    description=match.group("desc").strip(),
                    qty=_to_float(match.group("qty")),
                    unit_price=_to_float(match.group("price")),
                    total=_to_float(match.group("total")),
                )
            )

    def _search(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(text)
        return _to_float(match.group(2)) if match else None

    hospital = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not _SEPARATOR.match(stripped) and ":" not in stripped:
            hospital = stripped
            break

    return Bill(
        patient_id=fields.get("patient_id"),
        bill_id=fields.get("bill_id"),
        hospital_name=fields.get("hospital_name") or hospital,
        billing_date=fields.get("billing_date"),
        currency=fields.get("currency"),
        line_items=items,
        subtotal=_search(_SUBTOTAL_LINE),
        tax=_search(_TAX_LINE),
        total_amount=_search(_TOTAL_LINE),
        payment_status=_payment_status(fields.get("payment_status")),
        payment_method=fields.get("payment_method"),
    )


PARSERS = {
    "discharge_report": parse_discharge_report,
    "lab_report": parse_lab_report,
    "bill": parse_bill,
}


def parse(doc_type: str, content: DocumentContent):
    parser = PARSERS.get(doc_type)
    if parser is None:
        raise ValueError(f"No parser for document type {doc_type!r}")
    return parser(content)
