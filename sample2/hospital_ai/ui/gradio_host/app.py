"""Host Orchestrator UI — Gradio, port 8083 (doc Table 15, Figure 1).

The operator-facing side of the Host Orchestrator: run the workflow, watch case
status and risk, inspect agent health, read the streaming summary, and query
the RAG agent.

The Streamlit dashboard (:8501) is where clinicians review and correct a case;
this is where operations watches the system run.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import configure_logging, get_logger
from hospital_ai.ui.service import DashboardService

_log = get_logger(__name__, component="gradio-host")

CSS = """
.gradio-container { max-width: 1240px !important; }
#df-hero { background: linear-gradient(135deg, #2b5fd9 0%, #1b3fa0 100%);
  color: #fff; padding: 20px 26px; border-radius: 16px; margin-bottom: 14px; }
#df-hero h1 { margin: 0; font-size: 22px; font-weight: 700; }
#df-hero p { margin: 4px 0 0; opacity: .88; font-size: 13px; }
.df-note { font-size: 12px; color: #64748b; }
"""


def _cases_frame(service: DashboardService) -> pd.DataFrame:
    rows = [
        {
            "Case": case["case_id"],
            "Patient": case["patient_id"],
            "Status": case["status"],
            "Risk": case.get("risk_level") or "—",
            "Score": case.get("risk_score"),
            "Blocked": bool(case.get("discharge_blocked")),
            "Updated": case["updated_at"],
        }
        for case in service.cases()
    ]
    return pd.DataFrame(rows or [{"Case": "—", "Patient": "—", "Status": "no cases yet"}])


def build_ui() -> gr.Blocks:
    settings = get_settings()
    configure_logging(settings.log_level, settings.reports_dir / "pipeline.log")
    service = DashboardService()

    with gr.Blocks(title="DischargeFlow — Host Orchestrator") as ui:
        gr.HTML(
            '<div id="df-hero"><h1>DischargeFlow — Host Orchestrator</h1>'
            "<p>Google ADK workflow controller · A2A client · "
            "LangGraph · Agno · dual MCP servers</p></div>"
        )

        with gr.Tab("Run workflow"):
            with gr.Row():
                patient = gr.Dropdown(label="Patient", choices=[], interactive=True, scale=3)
                refresh = gr.Button("🔄 Scan workspace", scale=1)
            with gr.Row():
                run_one = gr.Button("▶ Process patient", variant="primary")
                run_all = gr.Button("▶▶ Process every patient")

            outcome_md = gr.Markdown()
            cases_table = gr.Dataframe(label="Cases", interactive=False, wrap=True)

            def scan() -> tuple[Any, str, pd.DataFrame]:
                discovered = service.discover()
                ids = [entry["patient_id"] for entry in discovered["patients"]]
                note = (
                    f"Found **{discovered['patient_count']} patients** and "
                    f"**{discovered['document_count']} documents** under "
                    f"`{', '.join(discovered['roots'])}`."
                )
                return gr.update(choices=ids, value=ids[0] if ids else None), note, _cases_frame(service)

            def process(patient_id: str) -> tuple[str, pd.DataFrame]:
                if not patient_id:
                    return "Select a patient first.", _cases_frame(service)
                outcome = service.process_patient(patient_id)
                verdict = (
                    f"⛔ **Requires human review** — {outcome['risk_level']} risk"
                    if outcome["requires_hitl"]
                    else f"✅ **Cleared for release** — {outcome['risk_level']} risk"
                )
                findings = "\n".join(
                    f"- `{f['rule_id']}` ({f['severity']}) — {f['message']}"
                    for f in outcome["findings"]
                ) or "- No findings."
                return (
                    f"### {outcome['case_id']}\n\n{verdict}  \n"
                    f"Score **{outcome['risk_score']}** · completeness "
                    f"**{outcome['completeness_score']}%** · "
                    f"{outcome['indexed_chunks']} chunks indexed\n\n"
                    f"**Findings**\n{findings}\n\n"
                    f"<span class='df-note'>Trace `{outcome['trace_id']}`</span>",
                    _cases_frame(service),
                )

            def process_all() -> tuple[str, pd.DataFrame]:
                outcomes = service.process_all()
                lines = [
                    f"- **{o['patient_id']}** → {o['status']} "
                    f"({o['risk_level']}, score {o['risk_score']})"
                    for o in outcomes
                ]
                return "### Batch complete\n\n" + "\n".join(lines), _cases_frame(service)

            refresh.click(scan, outputs=[patient, outcome_md, cases_table])
            run_one.click(process, inputs=patient, outputs=[outcome_md, cases_table])
            run_all.click(process_all, outputs=[outcome_md, cases_table])
            ui.load(scan, outputs=[patient, outcome_md, cases_table])

        with gr.Tab("Streaming summary"):
            gr.Markdown(
                "Summaries stream section by section: patient → medications → "
                "labs → bill → instructions. Blocked cases are refused."
            )
            case_box = gr.Dropdown(label="Case", choices=[], interactive=True)
            load_cases = gr.Button("Refresh cases")
            stream_btn = gr.Button("▶ Stream summary", variant="primary")
            summary_md = gr.Markdown()

            def case_choices():
                return gr.update(choices=[c["case_id"] for c in service.cases()])

            def stream(case_id: str):
                if not case_id:
                    yield "Select a case first."
                    return
                text = ""
                for event in service.summary_events(case_id):
                    if event["type"] == "blocked":
                        yield f"⛔ **Blocked** — {event['message']}"
                        return
                    if event["type"] == "section":
                        text += f"### {event['title']}\n\n{event['content']}\n\n"
                        yield text

            load_cases.click(case_choices, outputs=case_box)
            stream_btn.click(stream, inputs=case_box, outputs=summary_md)

        with gr.Tab("Clinical Q&A"):
            gr.Markdown(
                "Grounded retrieval over indexed discharge records. Available for "
                "every indexed case, including ones awaiting human review."
            )
            question = gr.Textbox(
                label="Question",
                placeholder="What medications was P1019 discharged on?",
            )
            ask = gr.Button("Ask", variant="primary")
            answer_md = gr.Markdown()

            def query(text: str) -> str:
                if not text.strip():
                    return "Type a question first."
                try:
                    result = service.ask(text)
                except Exception as exc:
                    from hospital_ai.ui.service import unwrap_exception_group

                    root = unwrap_exception_group(exc)
                    return (
                        f"Clinical Q&A could not complete: {root}. "
                        "Process a patient first so records are indexed."
                    )
                triad = result["triad"]
                sources = "\n".join(
                    f"- `{c['patient_id']}` {c['doc_type']}/{c['section']} "
                    f"(score {c['score']:.3f})"
                    for c in result["chunks"]
                ) or "- No sources retrieved."
                banner = (
                    f"> 🛡 **Blocked** — {result['block_reason']}\n\n"
                    if result["blocked"]
                    else ""
                )
                return (
                    f"{banner}{result['answer']}\n\n"
                    f"**RAG Triad** — faithfulness {triad['faithfulness']:.2f} · "
                    f"answer relevance {triad['answer_relevance']:.2f} · "
                    f"context relevance {triad['context_relevance']:.2f}\n\n"
                    f"**Sources**\n{sources}"
                )

            ask.click(query, inputs=question, outputs=answer_md)

        with gr.Tab("System health"):
            health_btn = gr.Button("Check all agents", variant="primary")
            health_table = gr.Dataframe(label="A2A agents", interactive=False)
            stats_md = gr.Markdown()

            def health() -> tuple[pd.DataFrame, str]:
                results = service.agent_health()
                frame = pd.DataFrame(
                    [
                        {
                            "Agent": name,
                            "Port": info["port"],
                            "Up": "🟢" if info.get("up") else "🔴",
                            "Streaming": info.get("streaming", "—"),
                        }
                        for name, info in results.items()
                    ]
                )
                stats = service.stats()
                return frame, (
                    f"**Cases** {stats['total_cases']} · "
                    f"**blocked** {stats['blocked_cases']} · "
                    f"**by status** {stats['by_status']} · "
                    f"**by risk** {stats['by_risk_level']}"
                )

            health_btn.click(health, outputs=[health_table, stats_md])

        with gr.Tab("Settings"):
            gr.Markdown(
                f"""
                **Ports** — EHR {settings.ports.ehr} · primary MCP
                {settings.ports.primary_mcp} · analytics MCP
                {settings.ports.analytics_mcp} · host {settings.ports.host} ·
                dashboard {settings.ports.dashboard}

                **Agents** — extractor {settings.ports.extractor} · validator
                {settings.ports.validator} · normalizer {settings.ports.normalizer} ·
                monitor {settings.ports.monitor} · summary {settings.ports.summary} ·
                RAG {settings.ports.rag}

                **Models** — primary `{settings.llm.primary_model}`, fallback
                `{settings.llm.fallback_model}`, embeddings
                `{settings.llm.embedding_model}`

                **Rules version** — `{settings.rules_version}`

                **Roots workspace** — `{settings.roots.uri}`

                **LangFuse** — {'connected to ' + settings.langfuse.host
                                if settings.langfuse.enabled
                                else 'not configured; tracing to data/reports/traces.jsonl'}

                Secrets are read from the environment and are never displayed here.
                """
            )

    return ui


def main() -> None:
    settings = get_settings()
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=settings.ports.host,
        quiet=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
