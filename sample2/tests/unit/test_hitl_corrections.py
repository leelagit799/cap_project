"""Unit tests for HITL correction helpers."""

from __future__ import annotations

from hospital_ai.ui.hitl_corrections import (
    corrections_from_elicitation,
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
