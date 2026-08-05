"""Patient Upload page — dynamic document ingestion UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.ingest.service import UploadService
from hospital_ai.ui.service import DashboardService
from hospital_ai.ui.streamlit_hitl import theme

_DOC_LABELS = {
    "discharge_report": ("Doctor report", "PDF, DOCX, TXT, PNG, JPEG, JPG, JSON"),
    "lab_report": ("Lab report", "PDF, PNG, JPEG, JPG, TXT, JSON"),
    "bill": ("Hospital bill", "PDF, PNG, JPEG, JPG, TXT, JSON"),
}

_DOC_ICONS = {
    "discharge_report": "🩺",
    "lab_report": "🧪",
    "bill": "🧾",
}

_ACCEPT = {
    "discharge_report": [".pdf", ".docx", ".txt", ".json", ".png", ".jpg", ".jpeg"],
    "lab_report": [".pdf", ".txt", ".json", ".png", ".jpg", ".jpeg"],
    "bill": [".pdf", ".txt", ".json", ".png", ".jpg", ".jpeg"],
}


@st.cache_resource
def upload_service() -> UploadService:
    return UploadService()


def _init_state() -> None:
    if "upload_patient_id" not in st.session_state:
        st.session_state.upload_patient_id = None


def page_upload(svc: DashboardService) -> None:
    _init_state()
    svc = upload_service()

    st.markdown(
        theme.masthead(
            "Patient Upload",
            "Register a new patient, upload discharge documents, and start the existing workflow.",
            "Upload workspace",
        ),
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([2, 1])
    with top_left:
        if st.button("➕ New patient", type="primary", use_container_width=False):
            created = svc.create_patient()
            st.session_state.upload_patient_id = created["patient_id"]
            st.toast(f"Allocated {created['patient_id']}", icon="🆔")
            st.rerun()

    patients = svc.list_patients()
    patient_ids = [row["patient_id"] for row in patients]
    if patient_ids:
        with top_right:
            selected = st.selectbox(
                "Open patient",
                patient_ids,
                index=patient_ids.index(st.session_state.upload_patient_id)
                if st.session_state.upload_patient_id in patient_ids
                else 0,
                key="upload-patient-picker",
            )
            st.session_state.upload_patient_id = selected

    patient_id = st.session_state.upload_patient_id
    if not patient_id:
        st.info("Click **New patient** to generate a patient ID, then upload the three required documents.")
        if patients:
            st.markdown("#### Recent uploads")
            st.dataframe(
                [
                    {
                        "Patient": row["patient_id"],
                        "Documents": row["document_count"],
                        "Complete": "Yes" if row["complete"] else "No",
                    }
                    for row in patients
                ],
                use_container_width=True,
                hide_index=True,
            )
        return

    detail = svc.get_patient(patient_id)
    if detail is None:
        st.error("Patient record not found.")
        return

    st.markdown(
        f'<div class="df-id-badge">🪪 Patient ID <span>{patient_id}</span></div>',
        unsafe_allow_html=True,
    )

    completeness = "Complete packet" if detail["complete"] else "Awaiting documents"
    st.markdown(
        theme.metrics_row(
            [
                ("Documents", len(detail["documents"]), "uploaded"),
                ("Packet", completeness, "doctor · lab · bill"),
                ("Folder", f"data/input/{patient_id}", "on disk"),
            ]
        ),
        unsafe_allow_html=True,
    )

    doctor_name = st.text_input(
        "Attending physician name (for doctor report filename)",
        value=detail.get("doctor_name") or "",
        placeholder="e.g. Dr Smith",
        help="Used to build names like P1025_DrSmith.pdf",
    )

    st.markdown("#### Upload documents")
    st.caption("Drag files into each card. Files are renamed automatically before storage.")

    columns = st.columns(3)
    for column, doc_type in zip(columns, ("discharge_report", "lab_report", "bill")):
        label, formats = _DOC_LABELS[doc_type]
        icon = _DOC_ICONS[doc_type]
        with column:
            st.markdown(
                f'<div class="df-upload-card">'
                f"<h4>{icon} {label}</h4>"
                f"<p>{formats}</p></div>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                f"Upload {label}",
                type=[ext.lstrip(".") for ext in _ACCEPT[doc_type]],
                key=f"upload-{patient_id}-{doc_type}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                replace = st.checkbox("Replace existing", key=f"replace-{patient_id}-{doc_type}")
                if st.button(f"Save {label}", key=f"save-{patient_id}-{doc_type}", use_container_width=True):
                    progress = st.progress(0, text="Uploading…")
                    try:
                        progress.progress(35, text="Validating file…")
                        svc.upload_document(
                            patient_id,
                            doc_type,
                            uploaded.name,
                            uploaded.getvalue(),
                            doctor_name=doctor_name if doc_type == "discharge_report" else None,
                            replace=replace,
                        )
                        progress.progress(100, text="Stored")
                        st.toast(f"{label} saved", icon="✅")
                        st.rerun()
                    except DischargeFlowError as exc:
                        progress.empty()
                        st.error(str(exc))

    st.markdown("#### Uploaded files")
    if not detail["documents"]:
        st.warning("No documents uploaded yet.")
    else:
        for doc in detail["documents"]:
            row_left, row_right = st.columns([4, 1])
            with row_left:
                st.markdown(
                    f'<div class="df-file-row">'
                    f'<span>{_DOC_ICONS.get(doc["doc_type"], "📄")} '
                    f'<strong>{doc["filename"]}</strong></span>'
                    f'<span class="df-pill">{doc["doc_type"].replace("_", " ")}</span>'
                    f'<span class="df-pill">{doc["size_bytes"] // 1024} KB</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with row_right:
                if st.button("Delete", key=f"del-{patient_id}-{doc['filename']}"):
                    svc.delete_document(patient_id, doc["filename"])
                    st.toast("Document deleted", icon="🗑️")
                    st.rerun()

    st.divider()
    process_left, process_right = st.columns([3, 1])
    with process_right:
        disabled = not detail["complete"]
        if st.button(
            "▶ Process patient",
            type="primary",
            disabled=disabled,
            use_container_width=True,
        ):
            progress = st.progress(0, text="Starting workflow…")
            try:
                progress.progress(20, text="Discovering documents…")
                progress.progress(45, text="Extracting & validating…")
                outcome = svc.process_patient(patient_id)
                progress.progress(100, text="Complete")
                if outcome.get("requires_hitl"):
                    st.error(
                        f"Case **{outcome['case_id']}** scored **{outcome.get('risk_level')}** "
                        "and needs human review. Open Validation Report."
                    )
                else:
                    st.success(
                        f"Case **{outcome['case_id']}** cleared at **{outcome.get('risk_level')}** risk."
                    )
                st.toast("Workflow finished", icon="✅")
            except DischargeFlowError as exc:
                progress.empty()
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - show orchestrator errors in UI
                progress.empty()
                st.error(f"Processing failed: {exc}")

    with process_left:
        if disabled:
            st.caption("Upload doctor report, lab report, and hospital bill to enable processing.")
        else:
            st.caption(
                "Processing uses the existing Host Orchestrator pipeline — "
                "Monitor → Extractor → Normalizer → Validator → Reporter → RAG."
            )
