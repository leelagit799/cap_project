"""P1 — Mock EHR service on :8050.

Assertions are anchored on the deliberate test-case mismatches documented in
``mock_ehr/data.py``, so a regression in the export or the API surfaces here.
"""

from __future__ import annotations

import pytest

from mock_ehr.export_json import MANDATED_FILES, export


class TestJSONExport:
    def test_exports_the_five_files_named_in_table_14(self, tmp_path):
        written = export(tmp_path)
        for dataset in MANDATED_FILES:
            assert written[dataset].is_file()
            assert written[dataset].stat().st_size > 0

    def test_export_round_trips_every_patient(self, tmp_path):
        import json

        from mock_ehr import data as source

        export(tmp_path)
        exported = json.loads((tmp_path / "patients.json").read_text(encoding="utf-8"))
        assert exported.keys() == source.PATIENTS.keys()
        assert len(exported) == 24


class TestPatientEndpoints:
    def test_health(self, ehr_client):
        body = ehr_client.get("/health").json()
        assert body["status"] == "ok"
        assert body["patients"] == 24

    def test_list_patients(self, ehr_client):
        patients = ehr_client.get("/patients").json()
        assert len(patients) == 24
        assert patients[0]["patient_id"] == "P1001"

    def test_filter_by_service_line(self, ehr_client):
        cardiology = ehr_client.get("/patients", params={"service_line": "Cardiology"}).json()
        assert {p["patient_id"] for p in cardiology} == {"P1003", "P1007"}

    def test_get_patient(self, ehr_client):
        patient = ehr_client.get("/patients/P1019").json()
        assert patient["patient_name"] == "Thomas Wright"
        assert patient["primary_dx"] == ["E11.9", "I10"]

    def test_unknown_patient_is_404(self, ehr_client):
        assert ehr_client.get("/patients/P9999").status_code == 404


class TestClinicalEndpoints:
    @pytest.mark.parametrize("patient_id", ["P1022", "P1024", "P1016", "P1021"])
    def test_penicillin_allergies_on_file(self, ehr_client, patient_id):
        """These four drive the allergy_contradiction_check in Table 4."""
        body = ehr_client.get(f"/patients/{patient_id}/allergies").json()
        assert "Penicillin" in body["allergies"]

    def test_patient_without_allergies(self, ehr_client):
        body = ehr_client.get("/patients/P1023/allergies").json()
        assert body["allergies"] == []

    def test_medications(self, ehr_client):
        meds = ehr_client.get("/patients/P1019/medications").json()
        assert {m["name"] for m in meds} == {
            "Metformin", "Lisinopril", "Atorvastatin", "Aspirin"
        }

    def test_spanish_medication_spellings_preserved(self, ehr_client):
        """P1020's EHR orders use Spanish spellings; canonicalization happens
        in the Normalizer, not here."""
        names = {m["name"] for m in ehr_client.get("/patients/P1020/medications").json()}
        assert "Metformina" in names and "Atorvastatina" in names

    def test_abnormal_lab_filter(self, ehr_client):
        all_labs = ehr_client.get("/patients/P1019/labs").json()
        abnormal = ehr_client.get(
            "/patients/P1019/labs", params={"abnormal_only": True}
        ).json()
        assert len(all_labs) == 2
        assert len(abnormal) == 2
        assert all(lab["abnormal"] for lab in abnormal)

    def test_p1007_has_an_unresolved_abnormal_lab(self, ehr_client):
        """BNP 845 with no action_in_ehr — the abnormal_lab_unresolved case."""
        labs = ehr_client.get("/patients/P1007/labs").json()
        bnp = next(lab for lab in labs if lab["test"] == "BNP")
        assert bnp["abnormal"] is True
        assert bnp["action_in_ehr"] == ""

    def test_care_plan(self, ehr_client):
        plan = ehr_client.get("/patients/P1019/care-plan").json()
        assert plan == {
            "followup_required": True,
            "speciality": "Endocrinology",
            "window_days": 30,
        }


class TestGuidelinesAndBundle:
    def test_guideline_lookup(self, ehr_client):
        guideline = ehr_client.get("/guidelines/E11.9").json()
        assert guideline["diagnosis"] == "Type 2 Diabetes Mellitus"
        assert "Metformin" in guideline["essential_meds"]

    def test_guideline_lookup_is_case_insensitive(self, ehr_client):
        assert ehr_client.get("/guidelines/j18.9").status_code == 200

    def test_unknown_guideline_is_404(self, ehr_client):
        assert ehr_client.get("/guidelines/Z99.9").status_code == 404

    def test_bundle_returns_every_validation_input(self, ehr_client):
        bundle = ehr_client.get("/patients/P1022/bundle").json()
        assert bundle["patient"]["patient_name"] == "Daan Bakker"
        assert bundle["allergies"] == ["Penicillin"]
        assert any(m["name"] == "Amoxicillin" for m in bundle["medications"])
        assert bundle["care_plan"]["speciality"] == "PCP"
        assert "J18.9" in bundle["guidelines"]

    def test_bundle_matches_the_individual_endpoints(self, ehr_client):
        bundle = ehr_client.get("/patients/P1019/bundle").json()
        assert bundle["allergies"] == ehr_client.get(
            "/patients/P1019/allergies"
        ).json()["allergies"]
        assert bundle["medications"] == ehr_client.get(
            "/patients/P1019/medications"
        ).json()
