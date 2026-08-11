"""HITL Dashboard — Streamlit, port 8501 (doc Table 13).

The five pages the specification fixes:

1. Document Viewer      — patient selector, tabbed documents, language badge,
                          structured preview, process trigger
2. Validation Report    — completeness score, cross-validation issues, risk
                          badge, recommendation, blocked indicator, trace link
3. HITL Corrections     — editable medication table, discharge decision,
                          save, re-run validation
4. RAG Q&A              — patient filter, injection indicator,
                          streaming display, source panel, RAG Triad metrics
5. Discharge Summary    — patient-friendly summary, prescription table,
                          colour-coded labs, JSON/HTML/PDF export, trace link

Run with ``streamlit run hospital_ai/ui/streamlit_hitl/app.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from hospital_ai.core.config import get_settings
from hospital_ai.rag.formatting import mask_pii
from hospital_ai.ui.hitl_corrections import (
    apply_medication_suggestion,
    medication_correction_suggestions,
    medication_corrections_if_changed,
    normalize_medication_rows,
)
from hospital_ai.ui.service import (
    DashboardService,
    unwrap_exception_group,
)
from hospital_ai.ui.streamlit_hitl import icons, theme
from hospital_ai.observability import langfuse_status
from hospital_ai.ui.streamlit_hitl.page_upload import page_upload
from hospital_ai.ui.streamlit_hitl.session_context import (
    activate_from_outcome,
    clear_active_context,
    get_active_context,
    pending_process_patient,
    refresh_active_from_case,
    render_active_panel,
    require_active_case,
    set_pending_process,
)

st.set_page_config(
    page_title="DischargeFlow — Clinical Review",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

NAV_ITEMS: list[tuple[str, str, str]] = [
    ("upload", "folder-open", "Patient Documents"),
    ("documents", "file-search", "Document Viewer"),
    ("validation", "clipboard-check", "Validation Report"),
    ("corrections", "shield-check", "HITL Corrections"),
    ("rag", "message-circle", "Clinical Q&A"),
    ("summary", "file-text", "Discharge Summary"),
]

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "hi": "Hindi",
    "de": "German", "fr": "French", "nl": "Dutch",
}


@st.cache_resource
def service() -> DashboardService:
    return DashboardService()


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="df-card"><h3>{title}</h3>{body}</div>', unsafe_allow_html=True)


def _resolve_patient_name(patient_id: str) -> str:
    try:
        from hospital_ai.ingest.service import UploadService

        detail = UploadService().get_patient(patient_id)
        if detail and detail.get("doctor_name"):
            return detail["doctor_name"]
    except Exception:  # noqa: BLE001 - optional ingest metadata
        pass
    return patient_id


def _execute_process(svc: DashboardService, patient_id: str) -> None:
    """Run the Host Orchestrator workflow and bind the active session context."""
    patient_name = _resolve_patient_name(patient_id)
    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="df-card">{theme.skeleton(4)}</div>', unsafe_allow_html=True
    )
    with st.spinner("Running extraction, normalization, validation and reporting…"):
        outcome = svc.process_patient(patient_id)
    placeholder.empty()

    record = svc.record(outcome["case_id"]) or {}
    discharge = record.get("discharge_report") or {}
    if discharge.get("patient_name"):
        patient_name = discharge["patient_name"]

    activate_from_outcome(outcome, patient_name=patient_name)

    if outcome["requires_hitl"]:
        st.error(
            f"Case {outcome['case_id']} scored **{outcome['risk_level']}** "
            "and needs human review. Open **Validation Report**."
        )
    else:
        st.success(
            f"Case {outcome['case_id']} cleared at **{outcome['risk_level']}** risk."
        )
    st.toast("Processing complete — active case set", icon="✅")


def render_sidebar(svc: DashboardService) -> str:
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "upload"

    with st.sidebar:
        st.markdown(
            f'<div class="df-brand">'
            f'<div class="df-brand-icon">{icons.svg("heart-pulse", size=20)}</div>'
            f"<div><h2>DischargeFlow</h2>"
            f"<p>St. Marian Regional Medical Center</p></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="df-nav-label">Workspace</div>', unsafe_allow_html=True)
        for page_key, icon_name, label in NAV_ITEMS:
            is_active = st.session_state.nav_page == page_key
            icon_col, btn_col = st.columns([0.12, 0.88], vertical_alignment="center")
            with icon_col:
                st.markdown(icons.svg(icon_name, size=16), unsafe_allow_html=True)
            with btn_col:
                if st.button(
                    label,
                    key=f"nav-{page_key}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.nav_page = page_key
                    st.rerun()

        st.divider()
        active = get_active_context()
        if active:
            st.markdown("**Active case**")
            st.markdown(
                f'<div class="df-active-case">'
                f"<strong>{active.get('patient_id')}</strong><br>"
                f"<span>{str(active.get('case_id', ''))[:22]}…</span><br>"
                f"<span>Status: {active.get('workflow_status')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No active processing session.")

        st.divider()
        stats = svc.stats()
        st.markdown("**Case load**")
        st.metric("Total cases", stats["total_cases"])
        st.metric("Awaiting review", stats["by_status"].get("HITL_PENDING", 0))
        st.metric("Blocked", stats["blocked_cases"])

        if stats["by_risk_level"]:
            st.markdown("**Risk mix**")
            st.bar_chart(pd.Series(stats["by_risk_level"], name="cases"), height=150)

        st.divider()
        with st.expander("System health"):
            if st.button("Check agents", use_container_width=True):
                with st.spinner("Polling A2A agents…"):
                    health = svc.agent_health()
                for name, info in health.items():
                    st.write(f"{'🟢' if info.get('up') else '🔴'} {name} · :{info['port']}")

            if st.button("Check LangFuse", use_container_width=True):
                with st.spinner("Verifying LangFuse credentials…"):
                    status = langfuse_status(probe=True)
                icon = {"connected": "🟢", "disabled": "🟡", "error": "🔴"}.get(
                    status["state"], "⚪"
                )
                st.write(f"{icon} LangFuse · {status['message']}")
                if not status["env_file_exists"]:
                    st.caption(f"Create `{status['env_file']}` from `.env.example`.")

        settings = get_settings()
        lf = langfuse_status()
        lf_label = {
            "connected": f"connected ({lf['base_url']})",
            "disabled": "offline (JSONL only)",
            "error": "auth error (JSONL only)",
        }.get(lf["state"], "unknown")
        st.caption(
            f"Rules `{settings.rules_version[:12]}…`\n\n"
            f"LangFuse {lf_label}"
        )
    return st.session_state.nav_page


# --- Page 1: Document Viewer -------------------------------------------------


def page_documents(svc: DashboardService) -> None:
    st.markdown(
        theme.masthead(
            "Document Viewer",
            "Inspect the incoming discharge packet and start a review case.",
            "Page 1 of 5",
        ),
        unsafe_allow_html=True,
    )

    with st.spinner("Scanning the Roots-authorised workspace…"):
        discovered = svc.discover()

    if not discovered.get("patients"):
        st.warning("No discharge packets found in the input workspace.")
        return

    st.markdown(
        theme.metrics_row(
            [
                ("Patients", discovered["patient_count"], "in workspace"),
                ("Documents", discovered["document_count"], "source files"),
                ("Roots", len(discovered["roots"]), "authorised"),
            ]
        ),
        unsafe_allow_html=True,
    )

    ids = [entry["patient_id"] for entry in discovered["patients"]]
    left, right = st.columns([3, 1])
    with left:
        patient_id = st.selectbox("Patient", ids, key="document-viewer-patient")
    entry = next(e for e in discovered["patients"] if e["patient_id"] == patient_id)

    with right:
        st.write("")
        process_disabled = not entry["complete"]
        if st.button(
            "Process Patient",
            type="primary",
            use_container_width=True,
            disabled=process_disabled,
        ):
            active = get_active_context()
            if active and active.get("patient_id") != patient_id:
                set_pending_process(patient_id)
            else:
                _execute_process(svc, patient_id)

    pending = pending_process_patient()
    if pending == patient_id:
        active = get_active_context()
        st.warning(
            "A patient case is currently being processed. Starting a new case will end "
            f"the current workflow for **{active.get('patient_id')}** "
            f"({active.get('case_id')}). Do you want to continue?"
        )
        confirm_left, confirm_right = st.columns(2)
        with confirm_left:
            if st.button("Yes, start new case", type="primary", key="confirm-new-case"):
                clear_active_context()
                set_pending_process(None)
                _execute_process(svc, patient_id)
        with confirm_right:
            if st.button("Cancel", key="cancel-new-case"):
                set_pending_process(None)
                st.rerun()

    if process_disabled:
        st.caption("Upload doctor report, lab report, and hospital bill before processing.")

    completeness = "complete" if entry["complete"] else "incomplete"
    pills = "".join(f'<span class="df-pill">{t}</span>' for t in entry["doc_types"])
    card(
        "Packet",
        f'{pills}<span class="df-pill">{completeness}</span>',
    )

    tabs = st.tabs(["Discharge report", "Lab report", "Bill"])
    for tab, doc_type in zip(tabs, ("discharge_report", "lab_report", "bill")):
        with tab:
            uri = entry["primary_documents"].get(doc_type)
            if uri is None:
                st.warning(f"No {doc_type.replace('_', ' ')} in this packet.")
                continue

            document = next(d for d in entry["documents"] if d["uri"] == uri)
            badges = [f'<span class="df-pill">{document["media_type"]}</span>']
            if document.get("ocr_sidecar"):
                badges.append('<span class="df-pill">OCR transcription</span>')
            st.markdown(
                f'<div>{"".join(badges)}<span class="df-pill">{uri}</span></div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Extracting document text…"):
                text = svc.document_text(uri)
            st.text_area(
                "Source text",
                text,
                height=340,
                key=f"doc-{patient_id}-{doc_type}",
            )

    record = None
    for case in svc.cases(patient_id=patient_id):
        record = svc.record(case["case_id"])
        if record:
            break

    if record:
        with st.expander("Structured data preview"):
            language = (record.get("source_language") or "en").lower()
            st.markdown(
                f'<span class="df-pill">Language detected: '
                f'{LANGUAGE_NAMES.get(language, language)}</span>',
                unsafe_allow_html=True,
            )
            st.json(record, expanded=False)


# --- Page 2: Validation Report ----------------------------------------------


def page_validation(svc: DashboardService) -> None:
    st.markdown(
        theme.masthead(
            "Validation Report",
            "Completeness, cross-validation findings and the discharge decision.",
            "Page 2 of 5",
        ),
        unsafe_allow_html=True,
    )

    case = require_active_case(svc)
    if case is None:
        return

    render_active_panel()

    validation = svc.validation(case["case_id"])
    if validation is None:
        st.warning("This case has no validation run yet.")
        return

    blocked = validation["discharge_blocked"]
    blocked_label = "Discharge blocked" if blocked else "Cleared for release"
    st.markdown(
        f'<div class="df-banner {"blocked" if blocked else "clear"}">'
        f"<strong>{blocked_label}</strong> — {validation['recommendation_text']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.metrics_row(
            [
                ("Risk level", validation["risk_level"], f"score {validation['risk_score']}"),
                ("Completeness", f"{validation['completeness_score']}%", "required + prescription fields"),
                (
                    "Translation",
                    f"{validation['translation_confidence']:.2f}"
                    if validation.get("translation_confidence") is not None
                    else "—",
                    "confidence",
                ),
                ("Findings", len(validation["findings"]), "raised"),
                ("Run", validation["run_no"], "validation pass"),
            ]
        ),
        unsafe_allow_html=True,
    )

    if validation.get("triggered_guardrails"):
        card(
            "Hard guardrails triggered",
            "".join(
                f'<span class="df-pill">{g}</span>'
                for g in validation["triggered_guardrails"]
            ),
        )

    findings = validation["findings"]
    if findings:
        frame = pd.DataFrame(
            [
                {
                    "Rule": f["rule_id"],
                    "Severity": f["severity"].upper(),
                    "Field": f.get("field") or "—",
                    "Detail": f["message"],
                    "Weight": f["weight"],
                    "Status": "Resolved" if f["resolved"] else ("Blocking" if f["blocking"] else "Open"),
                }
                for f in findings
            ]
        )
        severities = st.multiselect(
            "Filter by severity",
            sorted(frame["Severity"].unique()),
            default=sorted(frame["Severity"].unique()),
        )
        st.dataframe(
            frame[frame["Severity"].isin(severities)],
            use_container_width=True,
            hide_index=True,
        )

        counts = frame["Severity"].value_counts()
        st.bar_chart(counts, height=200)
    else:
        st.success("No findings. Every completeness and cross-validation rule passed.")

    reports = svc.report_paths(case["case_id"])
    columns = st.columns(4)
    for column, fmt in zip(columns, ("json", "html", "pdf")):
        path = reports.get(fmt)
        with column:
            if path is None:
                st.button(f"{fmt.upper()} unavailable", disabled=True, use_container_width=True)
            else:
                st.download_button(
                    f"⬇ Audit {fmt.upper()}",
                    path.read_bytes(),
                    file_name=f"{case['case_id']}-audit.{fmt}",
                    use_container_width=True,
                )
    with columns[3]:
        url = svc.trace_url(case["trace_id"])
        if url:
            st.link_button("🔍 LangFuse trace", url, use_container_width=True)
        else:
            st.caption(f"Trace `{case['trace_id'][:16]}…`")

    with st.expander("Audit trail"):
        trail = svc.audit_trail(case["case_id"])
        st.dataframe(pd.DataFrame(trail), use_container_width=True, hide_index=True)

    with st.expander("Raw validation JSON"):
        st.json(validation, expanded=False)


# --- Page 3: HITL Corrections -----------------------------------------------


def page_corrections(svc: DashboardService) -> None:
    st.markdown(
        theme.masthead(
            "HITL Corrections",
            "Correct the record, choose a discharge decision, and save or re-run validation.",
            "Page 3 of 5",
        ),
        unsafe_allow_html=True,
    )

    case = require_active_case(svc)
    if case is None:
        return

    render_active_panel()

    record = svc.record(case["case_id"])
    if record is None:
        st.warning("This case has no stored clinical record.")
        return

    discharge = record.get("discharge_report") or {}
    bill = record.get("bill") or {}
    corrections: dict[str, Any] = {}
    validation = svc.validation(case["case_id"]) or {}

    pending_key = f"pending-med-corrections-{case['case_id']}"
    stored_medications = normalize_medication_rows(discharge.get("medications") or [])
    if pending_key not in st.session_state:
        st.session_state[pending_key] = None
    elif (
        st.session_state[pending_key] is not None
        and normalize_medication_rows(st.session_state[pending_key]) == stored_medications
    ):
        st.session_state[pending_key] = None

    st.markdown("#### Medications")
    source_medications = st.session_state[pending_key] or stored_medications
    med_columns = [
        "sl_no", "medicine_name", "strength", "dosage", "frequency", "route",
        "period", "remarks", "total_quantity",
    ]
    medications = pd.DataFrame(source_medications)
    if medications.empty:
        medications = pd.DataFrame(columns=med_columns)
    else:
        for column in med_columns:
            if column not in medications.columns:
                medications[column] = None
    edited = st.data_editor(
        medications[med_columns],
        use_container_width=True,
        num_rows="dynamic",
        key=f"medication-editor-{case['case_id']}",
        column_config={
            "medicine_name": st.column_config.TextColumn("Medicine name", required=True),
        },
    )
    normalized = normalize_medication_rows(edited.to_dict("records"))
    corrections.update(medication_corrections_if_changed(stored_medications, normalized))
    if corrections.get("discharge_report.medications"):
        st.session_state[pending_key] = corrections["discharge_report.medications"]
        st.caption("Medication table has unsaved edits.")

    suggestions = medication_correction_suggestions(
        validation.get("findings") or [],
        normalized,
    )
    if suggestions:
        st.markdown("#### Medication correction suggestions")
        st.caption(
            "Based on the validation report. Apply a suggestion or edit the table above manually."
        )
        for index, suggestion in enumerate(suggestions):
            columns = st.columns([4, 1])
            with columns[0]:
                st.info(f"**{suggestion['title']}**\n\n{suggestion['detail']}")
            with columns[1]:
                action = suggestion.get("action")
                if action and st.button(
                    "Apply",
                    key=f"med-suggestion-{case['case_id']}-{index}",
                    use_container_width=True,
                ):
                    current_rows = corrections.get(
                        "discharge_report.medications",
                        normalized,
                    )
                    updated = apply_medication_suggestion(current_rows, action)
                    st.session_state[pending_key] = updated
                    st.toast("Medication suggestion applied", icon="💊")
                    st.rerun()
    elif validation.get("findings"):
        st.caption("No medication-specific correction suggestions for this validation run.")

    st.markdown("#### Demographics and billing")
    left, right = st.columns(2)
    with left:
        address = st.text_input("Address", discharge.get("address") or "")
        if address and address != (discharge.get("address") or ""):
            corrections["discharge_report.address"] = address

        age = st.text_input("Age", str(discharge.get("age") or ""))
        if age and age != str(discharge.get("age") or ""):
            corrections["discharge_report.age"] = int(age) if age.isdigit() else age

        physician = st.text_input(
            "Attending / approving physician",
            discharge.get("attending_physician")
            or discharge.get("discharge_approved_by")
            or "",
            placeholder="e.g. Dr. van Dijk, MD",
        )
        current_physician = discharge.get("attending_physician") or discharge.get("discharge_approved_by") or ""
        if physician and physician != current_physician:
            corrections["discharge_report.attending_physician"] = physician
            corrections["discharge_report.discharge_approved_by"] = physician
            corrections["discharge_report.discharge_approved"] = True
        elif physician and not discharge.get("discharge_approved"):
            corrections["discharge_report.discharge_approved"] = True

    with right:
        options = ["PAID", "UNPAID", "PARTIAL", "INSURANCE_GUARANTEED", "UNKNOWN"]
        current = bill.get("payment_status", "UNKNOWN")
        status = st.selectbox(
            "Payment status", options, index=options.index(current) if current in options else 4
        )
        if status != current:
            corrections["bill.payment_status"] = status

        existing_follow_up = discharge.get("follow_up_appointments") or []
        follow_up_default = existing_follow_up[0] if existing_follow_up else ""
        follow_up = st.text_input(
            "Follow-up appointment",
            value=follow_up_default,
            placeholder="e.g. Endocrinology 2026-07-02 with Dr. Kapoor",
        )
        if follow_up != follow_up_default:
            corrections["discharge_report.follow_up_appointments"] = [follow_up] if follow_up else []

    st.markdown("#### Decision")
    decision = st.segmented_control(
        "Decision",
        options=["allow", "reject", "no call"],
        default="no call",
        help=(
            "**allow** — discharge the patient regardless of validation findings. "
            "**reject** — deny discharge. "
            "**no call** — re-run automated validation and follow its outcome."
        ),
    )

    save, rerun = st.columns(2)
    with save:
        if st.button("Save feedback", use_container_width=True):
            outcome = svc.apply_hitl_decision(
                case["case_id"],
                decision,
                corrections=corrections,
                rerun_validation=False,
            )
            st.session_state[pending_key] = None
            refreshed = st.session_state.setdefault("rag_refresh_patients", [])
            if case["patient_id"] not in refreshed:
                refreshed.append(case["patient_id"])
            st.toast("Review saved", icon="💾")
            if decision == "no call":
                st.success("Feedback recorded and corrections saved to the case record.")
            elif decision == "allow":
                if outcome.get("summary_generated"):
                    st.success("Discharge approved and patient summary generated.")
                else:
                    st.success("Discharge approved. The patient may proceed to Discharge Summary.")
                    if outcome.get("summary_error"):
                        st.warning(outcome["summary_error"])
            else:
                st.success("Discharge denied. The case remains blocked for review.")
            refresh_active_from_case(svc.case(case["case_id"]) or case)
            st.rerun()

    with rerun:
        rerun_label = (
            "Re-run validation"
            if decision == "no call"
            else "Apply decision"
        )
        if st.button(rerun_label, type="primary", use_container_width=True):
            with st.spinner(
                "Re-running validation…" if decision == "no call" else "Applying decision…"
            ):
                outcome = svc.apply_hitl_decision(
                    case["case_id"],
                    decision,
                    corrections=corrections,
                    rerun_validation=True,
                )
            st.session_state[pending_key] = None
            refreshed = st.session_state.setdefault("rag_refresh_patients", [])
            if case["patient_id"] not in refreshed:
                refreshed.append(case["patient_id"])
            refresh_active_from_case(svc.case(case["case_id"]) or case)
            st.toast("Decision applied", icon="🔄")
            if decision == "allow":
                if outcome.get("summary_generated"):
                    st.success("Discharge approved and patient summary generated.")
                else:
                    st.success("Discharge approved by clinician override.")
                    if outcome.get("summary_error"):
                        st.warning(outcome["summary_error"])
            elif decision == "reject":
                st.error("Discharge denied by clinician override.")
            elif outcome.get("requires_hitl"):
                st.error(
                    f"Still requires review — {outcome.get('risk_level')} risk, "
                    f"score {outcome.get('risk_score')}."
                )
            else:
                st.success(
                    f"Validation cleared — {outcome.get('risk_level')} risk, "
                    f"score {outcome.get('risk_score')}."
                )
            st.rerun()

    runs = svc.validation_runs(case["case_id"])
    if len(runs) > 1:
        with st.expander(f"Validation history ({len(runs)} runs)"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Run": r["run_no"],
                            "Risk": r["risk_level"],
                            "Score": r["risk_score"],
                            "Blocked": r["discharge_blocked"],
                            "Findings": len(r["findings"]),
                            "At": r["created_at"],
                        }
                        for r in runs
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    reviews = svc.reviews(case["case_id"])
    if reviews:
        with st.expander(f"Previous reviews ({len(reviews)})"):
            st.dataframe(pd.DataFrame(reviews), use_container_width=True, hide_index=True)


# --- Page 4: RAG Q&A ---------------------------------------------------------


def page_rag(svc: DashboardService) -> None:
    st.markdown(
        theme.masthead(
            "Clinical Q&A",
            "Ask questions grounded in the indexed discharge records.",
            "Page 4 of 5",
        ),
        unsafe_allow_html=True,
    )

    cases = svc.cases()
    active = get_active_context()
    patients = sorted({c["patient_id"] for c in cases})
    if not patients:
        st.info("No cases indexed yet. Process a patient from Document Viewer first.")
        return

    default_patient = active["patient_id"] if active and active["patient_id"] in patients else "All patients"
    left, right = st.columns([1, 3])
    with left:
        options = ["All patients", *patients]
        default_index = options.index(default_patient) if default_patient in options else 0
        patient_filter = st.selectbox("Patient filter", options, index=default_index)
    patient_id = None if patient_filter == "All patients" else patient_filter
    active_case_id = None
    if active and patient_id and active.get("patient_id") == patient_id:
        active_case_id = active.get("case_id")
    elif active and not patient_id:
        active_case_id = active.get("case_id")
        patient_id = active.get("patient_id")

    if patient_id:
        refresh_patients = st.session_state.setdefault("rag_refresh_patients", [])
        if patient_id in refresh_patients:
            case_id = active_case_id or (
                svc.cases(patient_id=patient_id)[0]["case_id"]
                if svc.cases(patient_id=patient_id)
                else None
            )
            if case_id:
                try:
                    svc.reindex_case(case_id)
                except KeyError:
                    pass
            refresh_patients.remove(patient_id)
            st.session_state["rag_history"] = []
            st.info(
                "HITL corrections were saved for this patient. "
                "The Q&A index has been refreshed — ask again for updated answers."
            )

    if "rag_question" not in st.session_state:
        st.session_state.rag_question = ""

    question = st.text_input(
        "Your question", value=st.session_state.rag_question, key="rag-input"
    )

    from hospital_ai.guardrails import GuardrailManager

    injection = GuardrailManager().check_prompt_injection(question)
    if injection.blocked:
        st.markdown(
            '<div class="df-banner blocked">🛡 Prompt injection indicator — '
            f"{injection.detail}</div>",
            unsafe_allow_html=True,
        )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving, augmenting, generating and reflecting…"):
            try:
                answer = svc.ask(
                    question,
                    patient_id=patient_id,
                    case_id=active_case_id,
                )
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                root = unwrap_exception_group(exc)
                st.error(
                    "Clinical Q&A could not complete. "
                    f"{root}. Process a patient first so records are indexed, "
                    "or start the full stack with `python run.py`."
                )
                return

        st.session_state.setdefault("rag_history", []).insert(0, answer)

    for answer in st.session_state.get("rag_history", [])[:6]:
        st.markdown(
            f'<div class="df-chat q"><strong>Q</strong> · {answer["question"]}</div>',
            unsafe_allow_html=True,
        )

        if answer["blocked"]:
            st.markdown(
                f'<div class="df-banner blocked">🛡 Response blocked — '
                f'{answer["block_reason"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(answer["answer"])

        triad = answer["triad"]
        st.markdown(
            theme.metrics_row(
                [
                    ("Faithfulness", f"{triad['faithfulness']:.3f}", "≥ 0.70 required"),
                    ("Answer relevance", f"{triad['answer_relevance']:.3f}", ""),
                    ("Context relevance", f"{triad['context_relevance']:.3f}", ""),
                ]
            ),
            unsafe_allow_html=True,
        )

        if answer["chunks"]:
            with st.expander(f"Source documents ({len(answer['chunks'])})"):
                for chunk in answer["chunks"]:
                    st.markdown(
                        f'<span class="df-pill">{chunk["patient_id"]}</span>'
                        f'<span class="df-pill">{chunk["doc_type"]}/{chunk["section"]}</span>'
                        f'<span class="df-pill">score {chunk["score"]:.3f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(mask_pii(chunk["text"][:600]))
        st.divider()


# --- Page 5: Discharge Summary ----------------------------------------------


def page_summary(svc: DashboardService) -> None:
    st.markdown(
        theme.masthead(
            "Discharge Summary",
            "Patient-friendly summary for cases cleared by the escalation gate.",
            "Page 5 of 5",
        ),
        unsafe_allow_html=True,
    )

    case = require_active_case(svc)
    if case is None:
        return

    render_active_panel()

    if case.get("discharge_blocked"):
        st.markdown(
            '<div class="df-banner blocked">⛔ This discharge is blocked. '
            'No patient-facing summary can be generated until a clinician '
            'resolves the findings on the Corrections page.</div>',
            unsafe_allow_html=True,
        )
        return

    existing = svc.summary(case["case_id"])

    if st.button("Generate summary" if not existing else "Regenerate", type="primary"):
        placeholder = st.container()
        with placeholder:
            skeleton = st.empty()
            skeleton.markdown(
                f'<div class="df-card">{theme.skeleton(5)}</div>', unsafe_allow_html=True
            )
            with st.spinner("Streaming the summary section by section…"):
                events = svc.summary_events(case["case_id"])
            skeleton.empty()

        for event in events:
            if event["type"] == "blocked":
                st.error(event["message"])
                return
        existing = svc.summary(case["case_id"])
        st.toast("Summary generated", icon="📄")

    if not existing:
        st.info("No summary generated yet for this case.")
        return

    for section in sorted(existing["sections"], key=lambda s: s["order"]):
        st.markdown(
            f'<div class="df-section"><h4>{section["title"]}</h4>'
            f'<p>{section["content"]}</p></div>',
            unsafe_allow_html=True,
        )

    record = svc.record(case["case_id"]) or {}
    discharge = record.get("discharge_report") or {}

    medications = discharge.get("medications") or []
    if medications:
        st.markdown("#### Your prescriptions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Medicine": m.get("medicine_name"),
                        "Strength": m.get("strength"),
                        "How much": m.get("dosage"),
                        "How often": m.get("frequency_expanded") or m.get("frequency"),
                        "How": m.get("route"),
                        "For how long": m.get("period"),
                        "Notes": m.get("remarks"),
                    }
                    for m in medications
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    tests = (record.get("lab_report") or {}).get("tests") or []
    if tests:
        st.markdown("#### Your lab results")
        frame = pd.DataFrame(
            [
                {
                    "Test": t.get("test"),
                    "Result": f"{t.get('value')} {t.get('unit') or ''}".strip(),
                    "Normal range": t.get("reference_range"),
                    "Status": "Outside range" if t.get("abnormal") else "Normal",
                }
                for t in tests
            ]
        )

        def colour(row: pd.Series) -> list[str]:
            tint = "rgba(198,47,42,.14)" if row["Status"] == "Outside range" else "rgba(18,128,92,.12)"
            return [f"background-color: {tint}"] * len(row)

        st.dataframe(frame.style.apply(colour, axis=1), use_container_width=True, hide_index=True)

    st.markdown("#### Export")
    columns = st.columns(4)
    with columns[0]:
        st.download_button(
            "⬇ Summary JSON",
            json.dumps(existing, indent=2, ensure_ascii=False),
            file_name=f"{case['case_id']}-summary.json",
            use_container_width=True,
        )
    reports = svc.report_paths(case["case_id"])
    for column, fmt in zip(columns[1:], ("json", "html", "pdf")):
        path = reports.get(fmt)
        with column:
            if path is None:
                st.button(f"Audit {fmt.upper()} n/a", disabled=True, use_container_width=True)
            else:
                st.download_button(
                    f"⬇ Audit {fmt.upper()}",
                    path.read_bytes(),
                    file_name=f"{case['case_id']}-audit.{fmt}",
                    use_container_width=True,
                )

    url = svc.trace_url(case["trace_id"])
    if url:
        st.link_button("🔍 View LangFuse trace", url)


def main() -> None:
    svc = service()
    page = render_sidebar(svc)
    {
        "upload": page_upload,
        "documents": page_documents,
        "validation": page_validation,
        "corrections": page_corrections,
        "rag": page_rag,
        "summary": page_summary,
    }[page](svc)


if __name__ == "__main__":
    main()
