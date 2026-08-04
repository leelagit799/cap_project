"""Visual language for the HITL dashboard.

Streamlit is the framework doc Table 14 mandates, so the enterprise look is
built with custom CSS on top of it rather than by swapping in a JS stack. All
colours are driven by CSS custom properties that follow the user's light or
dark preference, so both modes come from one definition.
"""

from __future__ import annotations

CSS = """
<style>
:root {
  --df-bg: #f5f7fb;
  --df-surface: #ffffff;
  --df-surface-2: #eef2f9;
  --df-ink: #16233d;
  --df-muted: #64748b;
  --df-line: #dde5f0;
  --df-accent: #2b5fd9;
  --df-low: #12805c;
  --df-medium: #b26a00;
  --df-high: #c62f2a;
  --df-shadow: 0 1px 2px rgba(16,32,64,.06), 0 8px 24px rgba(16,32,64,.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --df-bg: #0d1220;
    --df-surface: #151d2e;
    --df-surface-2: #1b2436;
    --df-ink: #e9eefb;
    --df-muted: #93a3c0;
    --df-line: #26314a;
    --df-accent: #6f97ff;
    --df-low: #34d399;
    --df-medium: #fbbf24;
    --df-high: #f87171;
    --df-shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
  }
}

.stApp { background: var(--df-bg); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

/* ---- Masthead ---- */
.df-masthead {
  display: flex; align-items: center; justify-content: space-between;
  gap: 20px; padding: 18px 24px; margin-bottom: 22px; border-radius: 16px;
  background: linear-gradient(135deg, var(--df-accent) 0%, #1b3fa0 100%);
  color: #fff; box-shadow: var(--df-shadow);
  animation: df-fade .4s ease both;
}
.df-masthead h1 { margin: 0; font-size: 21px; font-weight: 700; letter-spacing: -.01em; }
.df-masthead p  { margin: 3px 0 0; font-size: 13px; opacity: .88; }
.df-masthead .df-page { font-size: 12px; text-transform: uppercase;
  letter-spacing: .12em; opacity: .85; }

/* ---- Cards ---- */
.df-card {
  background: var(--df-surface); border: 1px solid var(--df-line);
  border-radius: 14px; padding: 18px 20px; margin-bottom: 16px;
  box-shadow: var(--df-shadow); animation: df-fade .35s ease both;
}
.df-card h3 {
  margin: 0 0 12px; font-size: 12px; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; color: var(--df-muted);
}

/* ---- Metrics ---- */
.df-metrics { display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.df-metric {
  background: var(--df-surface); border: 1px solid var(--df-line);
  border-radius: 12px; padding: 14px 16px; transition: transform .18s ease;
}
.df-metric:hover { transform: translateY(-2px); }
.df-metric .k { font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--df-muted); }
.df-metric .v { font-size: 26px; font-weight: 700; color: var(--df-ink); margin-top: 2px; }
.df-metric .s { font-size: 12px; color: var(--df-muted); }

/* ---- Badges ---- */
.df-badge {
  display: inline-block; padding: 5px 13px; border-radius: 999px;
  font-size: 12px; font-weight: 700; letter-spacing: .04em; color: #fff;
}
.df-badge.Low { background: var(--df-low); }
.df-badge.Medium { background: var(--df-medium); }
.df-badge.High { background: var(--df-high); }
.df-badge.neutral { background: var(--df-muted); }

.df-pill {
  display: inline-block; padding: 3px 10px; border-radius: 8px; font-size: 11px;
  font-weight: 600; letter-spacing: .04em; border: 1px solid var(--df-line);
  background: var(--df-surface-2); color: var(--df-muted); margin-right: 6px;
}

/* ---- Banners ---- */
.df-banner { padding: 14px 18px; border-radius: 12px; font-weight: 600;
  margin-bottom: 16px; animation: df-fade .3s ease both; }
.df-banner.blocked { background: rgba(198,47,42,.12); color: var(--df-high);
  border: 1px solid rgba(198,47,42,.35); }
.df-banner.clear { background: rgba(18,128,92,.12); color: var(--df-low);
  border: 1px solid rgba(18,128,92,.32); }
.df-banner.info { background: rgba(43,95,217,.10); color: var(--df-accent);
  border: 1px solid rgba(43,95,217,.28); }

/* ---- Severity text ---- */
.sev-critical { color: var(--df-high); font-weight: 700; }
.sev-warning  { color: var(--df-medium); font-weight: 700; }
.sev-info     { color: var(--df-muted); font-weight: 600; }

/* ---- Streaming summary ---- */
.df-section {
  border-left: 3px solid var(--df-accent); padding: 4px 0 4px 16px;
  margin-bottom: 18px; animation: df-slide .4s ease both;
}
.df-section h4 { margin: 0 0 6px; font-size: 15px; color: var(--df-ink); }
.df-section p { margin: 0; color: var(--df-muted); line-height: 1.62; }

/* ---- Chat ---- */
.df-chat { border-radius: 12px; padding: 13px 16px; margin-bottom: 11px;
  animation: df-fade .3s ease both; }
.df-chat.q { background: var(--df-surface-2); border: 1px solid var(--df-line); }
.df-chat.a { background: var(--df-surface); border: 1px solid var(--df-line);
  border-left: 3px solid var(--df-accent); }

/* ---- Skeleton loader ---- */
.df-skeleton { height: 14px; border-radius: 7px; margin: 8px 0;
  background: linear-gradient(90deg, var(--df-surface-2) 25%,
    var(--df-line) 37%, var(--df-surface-2) 63%);
  background-size: 400% 100%; animation: df-shimmer 1.3s ease-in-out infinite; }

@keyframes df-shimmer { 0% { background-position: 100% 50%; }
                        100% { background-position: 0 50%; } }
@keyframes df-fade  { from { opacity: 0; transform: translateY(6px); }
                      to { opacity: 1; transform: none; } }
@keyframes df-slide { from { opacity: 0; transform: translateX(-8px); }
                      to { opacity: 1; transform: none; } }

/* ---- Streamlit chrome ---- */
section[data-testid="stSidebar"] { background: var(--df-surface);
  border-right: 1px solid var(--df-line); }
div[data-testid="stDataFrame"] { border: 1px solid var(--df-line); border-radius: 10px; }
.stButton > button { border-radius: 9px; font-weight: 600; transition: transform .12s ease; }
.stButton > button:hover { transform: translateY(-1px); }
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
