"""Deterministic clinical translation for offline / demo mode.

Live Bedrock inference handles translation in production. This module provides
phrase-level replacements so the pipeline produces English canonical records
when ``LLM_OFFLINE=1`` or credentials are absent.
"""

from __future__ import annotations

import re

#: Longest phrases first within each language block.
_PHRASES: dict[str, tuple[tuple[str, str], ...]] = {
    "es": (
        ("Diabetes Mellitus Tipo 2", "Type 2 Diabetes Mellitus"),
        ("Hipertensión Esencial", "Essential Hypertension"),
        ("Sin alergias conocidas", "No known allergies"),
        ("Medicina General", "General Medicine"),
        ("Masculino", "Male"),
        ("Femenino", "Female"),
        ("resumen de alta", "discharge summary"),
        ("diagnóstico de alta", "discharge diagnosis"),
        ("instrucciones de alta", "discharge instructions"),
        ("recetas de alta", "discharge prescriptions"),
        ("cita de seguimiento", "follow-up appointment"),
        ("Continuar con dieta para diabéticos", "Continue diabetic diet"),
        ("caminar 30 minutos 5 veces por semana", "walk 30 minutes 5 times per week"),
        ("Automonitoreo diario de presión arterial y glucosa", "Daily self-monitoring of blood pressure and glucose"),
        ("registrar las lecturas", "record readings"),
        ("Tomar todos los medicamentos exactamente como se recetaron", "Take all medications exactly as prescribed"),
        ("Regrese a urgencias por dolor de pecho", "Return to emergency for chest pain"),
        ("dificultad respiratoria grave", "severe difficulty breathing"),
        ("Consejería farmacéutica completada", "Pharmacy counselling completed"),
        ("Clínica de Endocrinología", "Endocrinology clinic"),
        ("Control con médico de cabecera", "Follow-up with primary care physician"),
        ("Con las comidas", "With meals"),
        ("Controlar la presión", "Monitor blood pressure"),
        ("Por la noche", "At night"),
        ("Cardioprotector", "Cardioprotective"),
        ("Metformina", "Metformin"),
        ("Atorvastatina", "Atorvastatin"),
        ("tableta", "tablet"),
        ("días", "days"),
        ("paciente", "patient"),
        ("alergias", "allergies"),
        ("medicamentos", "medications"),
        ("seguimiento", "follow-up"),
        ("alta", "discharge"),
    ),
    "nl": (
        ("Buiten het ziekenhuis opgelopen pneumonie", "Community-acquired pneumonia"),
        ("Penicilline (gedocumenteerde huiduitslag)", "Penicillin (documented skin rash)"),
        ("Maak de antibioticakuur volledig af", "Complete the full antibiotic course"),
        ("Drink voldoende en neem voldoende rust", "Drink plenty of fluids and get adequate rest"),
        ("Keer terug naar de SEH bij hoge koorts", "Return to the emergency department for high fever"),
        ("kortademigheid of pijn op de borst", "shortness of breath or chest pain"),
        ("Controle bij de huisarts", "Follow-up with the general practitioner"),
        ("Volledige kuur afmaken", "Complete the full course"),
        ("Bij koorts", "For fever"),
        ("Algemene Geneeskunde", "General Medicine"),
        ("Man", "Male"),
        ("Vrouw", "Female"),
        ("ontslagbrief", "discharge letter"),
        ("ontslagdiagnose", "discharge diagnosis"),
        ("ontslaginstructies", "discharge instructions"),
        ("ontslagrecepten", "discharge prescriptions"),
        ("vervolgafspraak", "follow-up appointment"),
        ("allergieën", "allergies"),
        ("patiënt", "patient"),
        ("medicatie", "medication"),
        ("ontslag", "discharge"),
        ("controle", "follow-up"),
        ("Amoxicilline", "Amoxicillin"),
        ("Paracetamol", "Paracetamol"),
        ("Geneesmiddel", "Medicine"),
        ("Sterkte", "Strength"),
        ("Dosering", "Dosage"),
        ("Frequentie", "Frequency"),
        ("Toedieningsweg", "Route"),
        ("Duur", "Duration"),
        ("Opmerkingen", "Remarks"),
        ("Totale hoeveelheid", "Total quantity"),
        ("tablet", "tablet"),
        ("dagen", "days"),
        ("diagnose", "diagnosis"),
        ("ziekenhuis", "hospital"),
    ),
    "hi": (
        ("टाइप 2 डायबिटीज मेलिटस", "Type 2 Diabetes Mellitus"),
        ("उच्च रक्तचाप", "Hypertension"),
        ("सामान्य चिकित्सा", "General Medicine"),
        ("पुरुष", "Male"),
        ("महिला", "Female"),
        ("1 गोली", "1 tablet"),
        ("भोजन के साथ", "With meals"),
        ("सुबह", "In the morning"),
        ("रात को सोते समय", "At bedtime"),
        (
            "रोज़ अपना रक्त शर्करा जाँचें। कम नमक और कम शर्करा वाला आहार लें। सभी दवाएँ नियमित रूप से लें। चक्कर आना, अत्यधिक प्यास या सीने में दर्द होने पर तुरंत ER लौटें।",
            "Check your blood sugar daily. Follow a low-salt, low-sugar diet. Take all medicines regularly. Return to the ER immediately for dizziness, excessive thirst, or chest pain.",
        ),
        ("डिस्चार्ज निदान", "discharge diagnosis"),
        ("डिस्चार्ज सारांश", "discharge summary"),
        ("रोगी", "patient"),
        ("दवाएँ", "medicines"),
        ("निदान", "diagnosis"),
    ),
    "de": (
        ("Entlassungsdiagnose", "discharge diagnosis"),
        ("Entlassungsanweisungen", "discharge instructions"),
        ("Medikamente", "medications"),
        ("Patient", "patient"),
        ("Diagnose", "diagnosis"),
        ("Allergie", "allergy"),
        ("Entlassung", "discharge"),
        ("Nachsorge", "follow-up"),
    ),
    "te": (
        ("రోగి", "patient"),
        ("విడుదల", "discharge"),
        ("నిర్ధారణ", "diagnosis"),
        ("మందులు", "medicines"),
    ),
}

#: Shared glossary applied after language-specific phrases.
_COMMON: tuple[tuple[str, str], ...] = (
    ("ontslagbrief", "discharge letter"),
    ("patiëntnummer", "patient number"),
    ("ontslagdiagnose", "discharge diagnosis"),
    ("ontslaginstructies", "discharge instructions"),
    ("resumen de alta", "discharge summary"),
)


def offline_translate_clinical(text: str, source_language: str | None = None) -> str:
    """Translate clinical free text to English without an LLM."""
    if not text or not text.strip():
        return text

    language = (source_language or "en").lower()
    if language == "en":
        return text

    result = text
    phrases = _PHRASES.get(language, ()) + _COMMON
    for source, target in sorted(phrases, key=lambda pair: len(pair[0]), reverse=True):
        result = re.sub(re.escape(source), target, result, flags=re.IGNORECASE)

    return result


def infer_source_language_from_prompt(system: str | None) -> str | None:
    if not system:
        return None
    match = re.search(r"from\s+(\w+)\s+into", system, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None
