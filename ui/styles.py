# ui/styles.py
# Clean, minimal CSS for recruiter-friendly light theme UI

APP_CSS = """
<style>
/* Base typography and body reset */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1E293B;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1200px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] * {
    color: #334155 !important;
}

/* Top header */
.app-header {
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #E2E8F0;
}

.app-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #185FA5;
    margin: 0;
    letter-spacing: -0.02em;
}

.app-subtitle {
    font-size: 0.92rem;
    color: #64748B;
    margin-top: 4px;
}

/* Metric KPI card */
.kpi-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: left;
    margin-bottom: 1rem;
}

.kpi-card .kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-card .kpi-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: #185FA5;
    line-height: 1.2;
    margin-top: 4px;
}

.kpi-card .kpi-sub {
    font-size: 0.76rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F172A;
    margin: 1rem 0 0.5rem 0;
}

.section-desc {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1rem;
}

/* Result status badges */
.badge-strong {
    background-color: #DCFCE7;
    color: #166534;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}

.badge-possible {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}

.badge-notmatch {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}

/* Chips for skills */
.chip-matched {
    background-color: #DCFCE7;
    color: #15803D;
    border: 1px solid #86EFAC;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-block;
    margin: 3px;
}

.chip-missing {
    background-color: #FEE2E2;
    color: #B91C1C;
    border: 1px solid #FCA5A5;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-block;
    margin: 3px;
}

/* Clean expander */
.streamlit-expanderHeader {
    font-weight: 500;
    font-size: 0.88rem;
    color: #334155 !important;
}

/* Tables styling */
.dataframe {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 0.86rem !important;
}
</style>
"""

# Chart theme for Plotly (clean blue and grey)
PLOT_THEME = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#334155", size=12),
    colorway=["#185FA5", "#475569", "#0284C7", "#94A3B8", "#0EA5E9", "#CBD5E1"],
)
