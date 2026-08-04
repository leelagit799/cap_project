"""Materialise the Mock EHR JSON data files.

Doc Table 14 specifies the Mock EHR as "FastAPI :8050 · 5 JSON data files
(patients, medications, allergies, labs, care_plans)", while the repository
ships the records as Python dicts in ``mock_ehr/data.py``. Rather than
hand-duplicating 24 patients, ``data.py`` stays the single source of truth and
this module generates the JSON files from it.

Run directly (``python -m mock_ehr.export_json``) or let the EHR service call
``ensure_exported()`` at startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mock_ehr import data as ehr_data

DATA_DIR = Path(__file__).resolve().parent / "data"

#: The five files named in doc Table 14.
MANDATED_FILES = ("patients", "medications", "allergies", "labs", "care_plans")

#: Auxiliary export beyond the mandated five; backs GET /guidelines/{icd10}.
AUXILIARY_FILES = ("guidelines",)


def _payloads() -> dict[str, Any]:
    return {
        "patients": ehr_data.PATIENTS,
        "medications": ehr_data.MED_ORDERS,
        "allergies": ehr_data.ALLERGIES,
        "labs": ehr_data.LABS,
        "care_plans": ehr_data.CARE_PLANS,
        "guidelines": ehr_data.GUIDELINES,
    }


def export(target_dir: Path | None = None, *, force: bool = True) -> dict[str, Path]:
    """Write each dataset to ``<target_dir>/<name>.json``."""
    directory = target_dir or DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, payload in _payloads().items():
        path = directory / f"{name}.json"
        if path.exists() and not force:
            written[name] = path
            continue
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written[name] = path
    return written


def ensure_exported(target_dir: Path | None = None) -> dict[str, Path]:
    """Export only the files that are missing. Safe to call on every startup."""
    return export(target_dir, force=False)


if __name__ == "__main__":
    results = export()
    for dataset, path in results.items():
        marker = "" if dataset in MANDATED_FILES else "  (auxiliary)"
        print(f"wrote {path.relative_to(path.parents[2])}{marker}")
