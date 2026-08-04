"""Document parser edge cases."""

from __future__ import annotations

from hospital_ai.documents.parsers import (
    DISCHARGE_LABELS,
    parse_header_fields,
    parse_discharge_report,
)
from hospital_ai.documents.loaders import DocumentContent


class TestPatientIdParsing:
    def test_mojibake_dutch_label_still_extracts_patient_id(self):
        """UTF-8 misread as Latin-1 turns ë into Ã«; the alias must still match."""
        header = [
            "PatiÃ«ntnummer:        P1024",
            "Naam:                 Bram de Vries",
        ]
        fields = parse_header_fields(header, DISCHARGE_LABELS)
        assert fields["patient_id"] == "P1024"

    def test_mojibake_discharge_text_parses_patient_id(self):
        text = (
            "PatiÃ«ntnummer:        P1024\n"
            "Naam:                 Bram de Vries\n"
            "----------------------------------------------------------------\n"
            "ONTSLAGDIAGNOSE\n"
            "----------------------------------------------------------------\n"
            "1. Pneumonie\n"
        )
        report = parse_discharge_report(DocumentContent(text=text, media_type="text/plain"))
        assert report.patient_id == "P1024"
