"""Patient Documents page — manage documents for static sample patients."""

from __future__ import annotations

import streamlit as st

from hospital_ai.core.errors import DischargeFlowError
from hospital_ai.ingest.models import DocType
from hospital_ai.ingest.static_samples import get_patient, list_sample_patients, update_patient_documents
from hospital_ai.ui.service import DashboardService
from hospital_ai.ui.streamlit_hitl import theme
from hospital_ai.ui.streamlit_hitl.session_context import clear_active_context, get_active_context
from mock_ehr.data import PATIENTS

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

_DOC_ORDER: tuple[DocType, ...] = ("discharge_report", "lab_report", "bill")


def _patient_label(patient_id: str) -> str:
    name = PATIENTS.get(patient_id, {}).get("patient_name")
    return f"{patient_id} — {name}" if name else patient_id


def _init_state() -> None:
    if "doc_mgmt_patient_id" not in st.session_state:
        st.session_state.doc_mgmt_patient_id = list_sample_patients()[0]
    if "doc_mgmt_save_message" not in st.session_state:
        st.session_state.doc_mgmt_save_message = None


def _reset_edit_state(patient_id: str) -> None:
    for doc_type in _DOC_ORDER:
        st.session_state.pop(f"doc-delete-{patient_id}-{doc_type}", None)
    st.session_state.pop(f"upload-doctor-{patient_id}", None)
    for doc_type in _DOC_ORDER:
        st.session_state.pop(f"upload-{patient_id}-{doc_type}", None)


def page_upload(svc: DashboardService) -> None:
    """Manage documents for existing static sample patients."""
    _init_state()

    st.markdown(
        theme.masthead(
            "Patient Documents",
            "Select a sample patient, review their three documents, and save edits "
            "without creating a new patient record.",
            "Static sample workspace",
        ),
        unsafe_allow_html=True,
    )

    if st.session_state.doc_mgmt_save_message:
        st.markdown(
            f'<div class="df-banner clear">✅ {st.session_state.doc_mgmt_save_message}</div>',
            unsafe_allow_html=True,
        )

    patient_ids = list_sample_patients()
    labels = {_patient_label(pid): pid for pid in patient_ids}
    label_options = list(labels.keys())
    current_id = st.session_state.doc_mgmt_patient_id
    current_label = _patient_label(current_id) if current_id in patient_ids else label_options[0]

    selected_label = st.selectbox(
        "Patient ID",
        label_options,
        index=label_options.index(current_label) if current_label in label_options else 0,
        key="doc-mgmt-patient-picker",
        help="Static sample patients shipped with the demo workspace (P1019–P1024).",
    )
    patient_id = labels[selected_label]
    if patient_id != st.session_state.doc_mgmt_patient_id:
        st.session_state.doc_mgmt_patient_id = patient_id
        st.session_state.doc_mgmt_save_message = None

    detail = get_patient(patient_id)
    if detail is None:
        st.error("Patient record not found.")
        return

    st.markdown(
        f'<div class="df-id-badge">🪪 Patient ID <span>{patient_id}</span></div>',
        unsafe_allow_html=True,
    )

    completeness = "Complete packet" if detail["complete"] else "Incomplete — upload every document"
    st.markdown(
        theme.metrics_row(
            [
                ("Documents", len(detail["documents"]), "primary files"),
                ("Packet", completeness, "doctor · lab · bill"),
                ("Workspace", "Data/incoming", "static samples"),
            ]
        ),
        unsafe_allow_html=True,
    )

    if not detail["complete"]:
        st.warning(
            "This patient is missing one or more documents. Upload replacements for every "
            "missing slot, then click **Save all changes**."
        )

    doctor_name = st.text_input(
        "Attending physician name (for doctor report filename)",
        value=detail.get("doctor_name") or "",
        placeholder="e.g. Dr Smith",
        help="Used when replacing the doctor report, e.g. P1019_DrSmith.pdf",
        key=f"upload-doctor-{patient_id}",
    )

    st.markdown("#### Patient documents")
    st.caption(
        "Review the three documents below. Mark a document for deletion, upload a replacement, "
        "then click **Save all changes** once every slot is accounted for."
    )

    saved_by_type = {doc["doc_type"]: doc for doc in detail["documents"]}
    pending_replacements: dict[DocType, object] = {}
    marked_delete: set[DocType] = set()

    columns = st.columns(3)
    for column, doc_type in zip(columns, _DOC_ORDER):
        label, formats = _DOC_LABELS[doc_type]
        icon = _DOC_ICONS[doc_type]
        with column:
            st.markdown(
                f'<div class="df-upload-card">'
                f"<h4>{icon} {label}</h4>"
                f"<p>{formats}</p></div>",
                unsafe_allow_html=True,
            )

            existing = saved_by_type.get(doc_type)
            if existing:
                st.markdown(
                    f'<div class="df-file-row">'
                    f'<span><strong>{existing["filename"]}</strong></span>'
                    f'<span class="df-pill">{existing["size_bytes"] // 1024} KB</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span class="df-pill">Not uploaded</span>', unsafe_allow_html=True)

            delete_key = f"doc-delete-{patient_id}-{doc_type}"
            marked = st.checkbox(
                "Delete document",
                key=delete_key,
                help="Remove this document when you save. You must upload a replacement.",
            )
            if marked:
                marked_delete.add(doc_type)
                st.markdown('<span class="df-pill">Marked for deletion</span>', unsafe_allow_html=True)

            uploaded = st.file_uploader(
                f"Replace {label}",
                type=[ext.lstrip(".") for ext in _ACCEPT[doc_type]],
                key=f"upload-{patient_id}-{doc_type}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                pending_replacements[doc_type] = uploaded
                st.markdown(
                    f'<span class="df-pill">Replacement: {uploaded.name}</span>',
                    unsafe_allow_html=True,
                )

    def _final_has_document(doc_type: DocType) -> bool:
        if doc_type in pending_replacements:
            return True
        if doc_type in marked_delete:
            return False
        return doc_type in saved_by_type

    packet_complete = all(_final_has_document(doc_type) for doc_type in _DOC_ORDER)
    has_changes = bool(marked_delete or pending_replacements)

    missing_labels = [
        _DOC_LABELS[doc_type][0]
        for doc_type in _DOC_ORDER
        if not _final_has_document(doc_type)
    ]

    save_col, _ = st.columns([1, 2])
    with save_col:
        if st.button(
            "Save all changes",
            type="primary",
            use_container_width=True,
            disabled=not packet_complete or not has_changes,
        ):
            progress = st.progress(0, text="Validating documents…")
            try:
                replacements: dict[DocType, tuple[str, bytes]] = {}
                for doc_type, uploaded in pending_replacements.items():
                    replacements[doc_type] = (uploaded.name, uploaded.getvalue())  # type: ignore[union-attr]

                progress.progress(35, text="Saving documents…")
                update_patient_documents(
                    patient_id,
                    replacements=replacements,
                    deleted_types=marked_delete,
                    doctor_name=doctor_name or None,
                )

                progress.progress(70, text="Clearing prior workflow results…")
                reset = svc.reset_patient_workflow(patient_id)
                active = get_active_context()
                if active and active.get("patient_id") == patient_id:
                    clear_active_context()

                progress.progress(100, text="Complete")
                cleared = len(reset["deleted_cases"])
                st.session_state.doc_mgmt_save_message = (
                    f"Documents saved for {patient_id}. "
                    f"Cleared {cleared} prior case(s) from the dashboard."
                )
                _reset_edit_state(patient_id)
                st.toast("Patient documents saved successfully.", icon="✅")
                st.rerun()
            except DischargeFlowError as exc:
                progress.empty()
                st.error(str(exc))

    if not packet_complete and missing_labels:
        st.caption(
            "Upload replacements for: "
            + ", ".join(missing_labels)
            + ". All three documents are required before saving."
        )
    elif packet_complete and not has_changes:
        st.caption("No pending edits. Mark a document for deletion or choose a replacement file to save.")
    elif packet_complete and has_changes:
        st.caption("Ready to save. Changes will update this patient only — no new Patient ID is created.")

    st.markdown("#### Current files on disk")
    all_docs = detail.get("all_documents") or detail["documents"]
    if not all_docs:
        st.warning("No documents saved yet.")
    else:
        for doc in all_docs:
            st.markdown(
                f'<div class="df-file-row">'
                f'<span>{_DOC_ICONS.get(doc["doc_type"], "📄")} '
                f'<strong>{doc["filename"]}</strong></span>'
                f'<span class="df-pill">{doc["doc_type"].replace("_", " ")}</span>'
                f'<span class="df-pill">{doc["size_bytes"] // 1024} KB</span>'
                f'<span class="df-pill">{doc["uri"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "After saving, open **Document Viewer**, select the same patient, and click "
        "**Process Patient** to run the AI workflow on the updated documents."
    )
