"""Visual language for the HITL dashboard — light mode only."""

from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --df-bg: #f8fafc;
  --df-surface: #ffffff;
  --df-surface-2: #f1f5f9;
  --df-ink: #0f172a;
  --df-muted: #64748b;
  --df-line: #e2e8f0;
  --df-accent: #2563eb;
  --df-accent-soft: rgba(37, 99, 235, 0.08);
  --df-low: #059669;
  --df-medium: #d97706;
  --df-high: #dc2626;
  --df-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 16px rgba(15, 23, 42, 0.06);
  --df-radius: 12px;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp { background: var(--df-bg); color: var(--df-ink); }
.block-container { padding-top: 1.5rem; max-width: 1280px; }

/* ---- Icons ---- */
.df-icon { display: inline-block; vertical-align: middle; flex-shrink: 0; }
.df-icon-lg { width: 22px; height: 22px; }

/* ---- Masthead ---- */
.df-masthead {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 20px; padding: 20px 24px; margin-bottom: 20px; border-radius: var(--df-radius);
  background: var(--df-surface); border: 1px solid var(--df-line);
  box-shadow: var(--df-shadow); animation: df-fade .35s ease both;
}
.df-masthead h1 {
  margin: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em;
  color: var(--df-ink);
}
.df-masthead p { margin: 4px 0 0; font-size: 0.875rem; color: var(--df-muted); line-height: 1.5; }
.df-masthead .df-page {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .1em; color: var(--df-muted); white-space: nowrap;
  padding: 6px 10px; border-radius: 8px; background: var(--df-surface-2);
  border: 1px solid var(--df-line);
}

/* ---- Sidebar brand & nav ---- */
.df-brand {
  display: flex; align-items: center; gap: 10px; padding: 4px 0 12px;
}
.df-brand-icon {
  display: flex; align-items: center; justify-content: center;
  width: 36px; height: 36px; border-radius: 10px;
  background: var(--df-accent-soft); color: var(--df-accent);
}
.df-brand h2 {
  margin: 0; font-size: 1rem; font-weight: 700; color: var(--df-ink); line-height: 1.2;
}
.df-brand p { margin: 2px 0 0; font-size: 11px; color: var(--df-muted); }

.df-nav-label {
  font-size: 10px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--df-muted); margin: 8px 0 6px;
}
.df-nav-row {
  display: flex; align-items: center; gap: 10px; margin-bottom: 2px;
}
.df-nav-row .df-icon { color: var(--df-muted); }

.df-active-case {
  background: var(--df-surface-2); border: 1px solid var(--df-line);
  border-radius: 10px; padding: 10px 12px; margin: 4px 0;
}
.df-active-case strong { font-size: 13px; color: var(--df-ink); }
.df-active-case span { font-size: 11px; color: var(--df-muted); }

/* ---- Cards ---- */
.df-card {
  background: var(--df-surface); border: 1px solid var(--df-line);
  border-radius: var(--df-radius); padding: 18px 20px; margin-bottom: 16px;
  box-shadow: var(--df-shadow); animation: df-fade .35s ease both;
}
.df-card h3 {
  margin: 0 0 12px; font-size: 11px; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: var(--df-muted);
}

/* ---- Metrics ---- */
.df-metrics { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
.df-metric {
  background: var(--df-surface); border: 1px solid var(--df-line);
  border-radius: 10px; padding: 14px 16px; transition: border-color .15s ease;
}
.df-metric:hover { border-color: #cbd5e1; }
.df-metric .k { font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--df-muted); font-weight: 600; }
.df-metric .v { font-size: 1.5rem; font-weight: 700; color: var(--df-ink); margin-top: 2px; }
.df-metric .s { font-size: 12px; color: var(--df-muted); margin-top: 2px; }

/* ---- Badges ---- */
.df-badge {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: 11px; font-weight: 700; letter-spacing: .03em; color: #fff;
}
.df-badge.Low { background: var(--df-low); }
.df-badge.Medium { background: var(--df-medium); }
.df-badge.High { background: var(--df-high); }
.df-badge.neutral { background: #94a3b8; }

.df-pill {
  display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 11px;
  font-weight: 500; border: 1px solid var(--df-line);
  background: var(--df-surface-2); color: var(--df-muted); margin-right: 6px; margin-bottom: 4px;
}

/* ---- Banners ---- */
.df-banner { padding: 12px 16px; border-radius: 10px; font-weight: 500; font-size: 14px;
  margin-bottom: 16px; animation: df-fade .3s ease both; line-height: 1.5; }
.df-banner.blocked { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.df-banner.clear { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.df-banner.info { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

/* ---- Severity text ---- */
.sev-critical { color: var(--df-high); font-weight: 600; }
.sev-warning  { color: var(--df-medium); font-weight: 600; }
.sev-info     { color: var(--df-muted); font-weight: 500; }

/* ---- Streaming summary ---- */
.df-section {
  border-left: 3px solid var(--df-accent); padding: 4px 0 4px 16px;
  margin-bottom: 18px; animation: df-slide .4s ease both;
}
.df-section h4 { margin: 0 0 6px; font-size: 15px; color: var(--df-ink); font-weight: 600; }
.df-section p { margin: 0; color: var(--df-muted); line-height: 1.65; }

/* ---- Chat ---- */
.df-chat { border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
  animation: df-fade .3s ease both; font-size: 14px; }
.df-chat.q { background: var(--df-surface-2); border: 1px solid var(--df-line); }
.df-chat.a { background: var(--df-surface); border: 1px solid var(--df-line);
  border-left: 3px solid var(--df-accent); }

/* ---- Skeleton loader ---- */
.df-skeleton { height: 12px; border-radius: 6px; margin: 8px 0;
  background: linear-gradient(90deg, var(--df-surface-2) 25%,
    var(--df-line) 37%, var(--df-surface-2) 63%);
  background-size: 400% 100%; animation: df-shimmer 1.3s ease-in-out infinite; }

@keyframes df-shimmer { 0% { background-position: 100% 50%; }
                        100% { background-position: 0 50%; } }
@keyframes df-fade  { from { opacity: 0; transform: translateY(4px); }
                      to { opacity: 1; transform: none; } }
@keyframes df-slide { from { opacity: 0; transform: translateX(-6px); }
                      to { opacity: 1; transform: none; } }

/* ---- Streamlit chrome ---- */
header[data-testid="stHeader"] {
  visibility: visible !important;
  display: block !important;
  background: transparent !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
  visibility: visible !important;
}
section[data-testid="stSidebar"] {
  background: var(--df-surface) !important;
  border-right: 1px solid var(--df-line);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--df-accent-soft) !important;
  color: var(--df-accent) !important;
  border: 1px solid rgba(37, 99, 235, 0.25) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid transparent !important;
  color: var(--df-ink) !important;
  font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  background: var(--df-surface-2) !important;
  border-color: var(--df-line) !important;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--df-line); border-radius: 10px; }
.stButton > button { border-radius: 8px; font-weight: 600; }
.stDownloadButton > button { border-radius: 8px; font-weight: 600; }

/* ---- Upload workspace ---- */
.df-upload-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.df-upload-card {
  background: var(--df-surface); border: 1px dashed var(--df-line); border-radius: var(--df-radius);
  padding: 16px; min-height: 72px; transition: border-color .2s ease;
}
.df-upload-card:hover { border-color: #94a3b8; }
.df-upload-card h4 {
  margin: 0; font-size: 14px; color: var(--df-ink); font-weight: 600;
  display: flex; align-items: center; gap: 8px;
}
.df-upload-card p { margin: 6px 0 0; font-size: 12px; color: var(--df-muted); }
.df-id-badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px;
  background: var(--df-accent-soft); color: var(--df-accent); font-weight: 600; font-size: 13px;
  border: 1px solid rgba(37, 99, 235, 0.2); margin-bottom: 16px;
}
.df-file-row {
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  padding: 10px 12px; border: 1px solid var(--df-line); border-radius: 8px; margin-bottom: 8px;
  background: var(--df-surface-2); font-size: 13px;
}
.df-file-row .df-file-name {
  display: flex; align-items: center; gap: 8px;
}
</style>
"""


def masthead(title: str, subtitle: str, page_label: str) -> str:
    return f"""
<div class="df-masthead">
  <div>
    <h1>{title}</h1>
    <p>{subtitle}</p>
  </div>
  <div class="df-page">{page_label}</div>
</div>
"""


def metric(label: str, value: object, sub: str = "") -> str:
    return (
        f'<div class="df-metric"><div class="k">{label}</div>'
        f'<div class="v">{value}</div>'
        + (f'<div class="s">{sub}</div>' if sub else "")
        + "</div>"
    )


def metrics_row(items: list[tuple[str, object, str]]) -> str:
    return (
        '<div class="df-metrics">'
        + "".join(metric(label, value, sub) for label, value, sub in items)
        + "</div>"
    )


def risk_badge(level: str | None) -> str:
    if not level:
        return '<span class="df-badge neutral">Not scored</span>'
    return f'<span class="df-badge {level}">{level} risk</span>'


def skeleton(lines: int = 3) -> str:
    return "".join(
        f'<div class="df-skeleton" style="width:{width}%"></div>'
        for width in ([92, 78, 85, 70, 88][:lines])
    )
