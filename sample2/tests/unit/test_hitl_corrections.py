"""Unit tests for HITL correction helpers."""

from __future__ import annotations

from hospital_ai.ui.hitl_corrections import (
    apply_medication_suggestion,
    corrections_from_elicitation,
    medication_correction_suggestions,
    merge_corrections,
    normalize_medication_rows,
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
        assert suggestions[0]["title"].startswith("Add 'metformin'")
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
