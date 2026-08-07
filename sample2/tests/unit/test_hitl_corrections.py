"""Unit tests for HITL correction helpers."""

from __future__ import annotations

from hospital_ai.ui.hitl_corrections import (
    apply_medication_suggestion,
    corrections_from_elicitation,
    medication_correction_suggestions,
    medication_corrections_if_changed,
    merge_corrections,
    normalize_medication_rows,
    sanitize_medicine_name,
)


class TestHitlCorrections:
    def test_elicitation_maps_doctors_to_physician(self):
        corrections = corrections_from_elicitation({"doctors": "Dr. van Dijk, MD"})
        assert corrections["discharge_report.attending_physician"] == "Dr. van Dijk, MD"
        assert corrections["discharge_report.discharge_approved_by"] == "Dr. van Dijk, MD"
        assert corrections["discharge_report.discharge_approved"] is True

    def test_elicitation_maps_address_and_follow_up(self):
        corrections = corrections_from_elicitation(
            {
                "address": "14 Lakeview Road, Mumbai",
                "follow_up_appointments": "Endocrinology 2026-07-02",
            }
        )
        assert corrections["discharge_report.address"] == "14 Lakeview Road, Mumbai"
        assert corrections["discharge_report.follow_up_appointments"] == [
            "Endocrinology 2026-07-02"
        ]

    def test_normalize_medication_rows_preserves_medicine_name(self):
        rows = normalize_medication_rows(
            [
                {
                    "sl_no": 1,
                    "name": "Metformin",
                    "strength": "500 mg",
                    "dosage": "1 tab",
                    "frequency": "BID",
                    "route": "ORAL",
                    "period": "30 days",
                    "remarks": None,
                    "total_quantity": "60",
                }
            ]
        )
        assert rows[0]["medicine_name"] == "Metformin"
        assert "name" not in rows[0]

    def test_merge_corrections_prefers_later_values(self):
        merged = merge_corrections(
            {"bill.payment_status": "UNPAID"},
            {"bill.payment_status": "PAID", "discharge_report.address": "Main Street"},
        )
        assert merged["bill.payment_status"] == "PAID"
        assert merged["discharge_report.address"] == "Main Street"

    def test_medication_suggestions_cover_omission_and_allergy(self):
        findings = [
            {
                "rule_id": "med_omission_check",
                "message": "'metformin' is on the EHR medication list but absent from discharge.",
                "expected": "metformin",
                "resolved": False,
            },
            {
                "rule_id": "allergy_contradiction_check",
                "message": "Amoxicilline is prescribed despite a documented Penicillin allergy.",
                "actual": "Amoxicilline",
                "resolved": False,
            },
        ]
        suggestions = medication_correction_suggestions(findings, [])
        assert suggestions[0]["title"].startswith("Add 'Metformin'")
        assert suggestions[1]["action"]["type"] == "remove_row"

    def test_apply_medication_suggestion_adds_and_fills_rows(self):
        rows = apply_medication_suggestion(
            [{"medicine_name": "Metformin", "strength": "500 mg"}],
            {"type": "add_row", "row": {"medicine_name": "Atorvastatin"}},
        )
        assert len(rows) == 2
        assert rows[-1]["medicine_name"] == "Atorvastatin"

        filled = apply_medication_suggestion(
            rows,
            {
                "type": "fill_row",
                "medicine_name": "Metformin",
                "fields": ["dosage", "period"],
            },
        )
        assert filled[0]["dosage"] == "As directed"
        assert filled[0]["period"] == "30 days"

    def test_medication_corrections_if_changed_detects_saved_record_diff(self):
        stored = [{"medicine_name": "Amoxicilline", "strength": "500 mg"}]
        current = [
            {"medicine_name": "Amoxicilline", "strength": "500 mg"},
            {"medicine_name": "Paracetamol", "strength": "500 mg"},
        ]
        payload = medication_corrections_if_changed(stored, current)
        assert "discharge_report.medications" in payload
        assert len(payload["discharge_report.medications"]) == 2

    def test_medication_corrections_if_changed_ignores_session_only_noise(self):
        rows = [{"medicine_name": "Metformin", "strength": "500 mg"}]
        assert medication_corrections_if_changed(rows, rows) == {}

    def test_sanitize_medicine_name_strips_description(self):
        assert sanitize_medicine_name(
            "Amoxicillin Amoxicillin is a medication. It is an antibiotic used to treat various bacterial infections."
        ) == "Amoxicillin"
        assert sanitize_medicine_name(
            "Paracetamol Paracetamol Paracetamol is a medication used to relieve pain"
        ) == "Paracetamol"
        assert sanitize_medicine_name("Azithromycin") == "Azithromycin"

    def test_normalize_medication_rows_sanitizes_names(self):
        rows = normalize_medication_rows(
            [
                {
                    "medicine_name": "Amoxicillin is a medication used for infections",
                    "strength": "500 mg",
                    "dosage": "1 tab",
                }
            ]
        )
        assert rows[0]["medicine_name"] == "Amoxicillin"
        assert rows[0]["strength"] == "500 mg"
