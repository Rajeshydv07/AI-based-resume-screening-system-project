import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(99,102,241,0.15);
}
.metric-card .val {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.78rem;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
}

.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #6366f1;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

.info-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    color: #c7d2fe;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

.top-header {
    background: linear-gradient(90deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 1.5rem;
}

hr { border-color: rgba(99,102,241,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join("data", "raw_resumes.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2rem;'>🎯</div>
        <div style='font-size:1.1rem; font-weight:800; color:#818cf8;'>RecruitAI</div>
        <div style='font-size:0.72rem; color:#64748b; margin-top:2px;'>AI Resume Screening Platform</div>
    </div>
    <hr style='margin-bottom:16px;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; font-weight:600; color:#6366f1; letter-spacing:.08em; margin-bottom:8px;'>📁 PROJECT MODULES</div>", unsafe_allow_html=True)

    menu = st.radio(
        "",
        [
            "📊  Dataset Explorer",
            "🔍  EDA & Insights",
            "🧹  Data Cleaning",
            "⚙️  Feature Engineering",
            "🤖  Model Training",
            "📈  Model Evaluation",
            "🚀  Full Pipeline",
            "🗂️  Resume Ranker",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; font-weight:600; color:#6366f1; letter-spacing:.08em; margin-bottom:8px;'>🗓️ PROJECT ROADMAP</div>", unsafe_allow_html=True)

    weeks = [
        ("Week 1", "Data Collection & EDA", "✅"),
        ("Week 2", "Cleaning & Preprocessing", "🔄"),
        ("Week 3", "Feature Engineering", "⏳"),
        ("Week 4", "Baseline Modelling", "⏳"),
        ("Week 5", "Hyperparameter Tuning", "⏳"),
        ("Week 6", "Model Evaluation", "⏳"),
        ("Week 7", "Pipeline & Deployment", "⏳"),
        ("Week 8", "Dashboard & Presentation", "⏳"),
    ]
    for w, title, status in weeks:
        color = "#4ade80" if status == "✅" else "#facc15" if status == "🔄" else "#475569"
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; padding:5px 0;'>"
            f"<span style='font-size:0.75rem;'>{status}</span>"
            f"<div><div style='font-size:0.72rem; color:{color}; font-weight:600;'>{w}</div>"
            f"<div style='font-size:0.68rem; color:#64748b;'>{title}</div></div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    df_sidebar = load_data()
    if df_sidebar is not None:
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.1); border-radius:10px; padding:10px 12px;'>
            <div style='font-size:0.68rem; color:#64748b;'>Dataset Loaded</div>
            <div style='font-size:0.9rem; font-weight:700; color:#818cf8;'>{len(df_sidebar):,} resumes</div>
            <div style='font-size:0.68rem; color:#64748b;'>{len(df_sidebar.columns)} features · raw_resumes.csv</div>
        </div>
        """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font_color="#cbd5e1",
    title_font_color="#e2e8f0",
    colorway=["#6366f1","#8b5cf6","#ec4899","#06b6d4","#10b981","#f59e0b"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
if "Dataset Explorer" in menu:
    st.markdown("""
    <div class='top-header'>
        <span style='font-size:1.6rem;'>📊</span>
        <b style='font-size:1.3rem; color:#e2e8f0;'> Dataset Explorer</b>
        <div style='font-size:0.8rem; color:#64748b; margin-top:4px;'>Week 1 — Raw data inspection, schema analysis &amp; quality summary</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("⚠️ Dataset not found. Run `python src/generate_data.py` first.")
        st.stop()

    # KPIs
    total_rows    = len(df)
    total_cols    = len(df.columns)
    missing_vals  = int(df.isnull().sum().sum())
    missing_pct   = round(missing_vals / (total_rows * total_cols) * 100, 1)
    duplicates    = int(df.duplicated().sum())
    shortlisted   = int(df["Shortlisted"].sum()) if "Shortlisted" in df.columns else 0
    shortlist_pct = round(shortlisted / total_rows * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, sub in [
        (c1, str(total_rows),    "Total Resumes",   "in dataset"),
        (c2, str(total_cols),    "Features",        "columns"),
        (c3, str(missing_vals),  "Missing Values",  f"{missing_pct}% of cells"),
        (c4, str(duplicates),    "Duplicates",      "rows to drop"),
        (c5, str(shortlisted),   "Shortlisted",     f"{shortlist_pct}% rate"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{val}</div>
            <div class='lbl'>{lbl}</div>
            <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Preview + Schema
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("<div class='section-header'>📋 Dataset Preview</div>", unsafe_allow_html=True)
        n = st.slider("Rows to display", 5, 50, 10, 5)
        st.dataframe(df.head(n), use_container_width=True, height=280)
    with right:
        st.markdown("<div class='section-header'>🗂️ Column Schema</div>", unsafe_allow_html=True)
        schema = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    [str(t) for t in df.dtypes],
            "Non-Null": df.notnull().sum().values,
            "Null %":   [f"{round(df[c].isnull().mean()*100,1)}%" for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True, height=280)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Target + Role dist
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("<div class='section-header'>🎯 Shortlisting Distribution</div>", unsafe_allow_html=True)
        counts = df["Shortlisted"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({1: "Shortlisted ✅", 0: "Rejected ❌"})
        fig = px.pie(counts, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={"Shortlisted ✅": "#4ade80", "Rejected ❌": "#f87171"},
                     hole=0.55)
        fig.update_layout(**CHART_THEME, height=300, margin=dict(t=10,b=10,l=10,r=10),
                          legend=dict(orientation="h", y=-0.1))
        fig.update_traces(textfont_size=13, textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>💼 Job Role Distribution</div>", unsafe_allow_html=True)
        role_counts = df["Job_Role"].value_counts().reset_index()
        role_counts.columns = ["Job_Role", "Count"]
        fig2 = px.bar(role_counts, x="Count", y="Job_Role", orientation="h",
                      color="Count", color_continuous_scale=["#312e81","#6366f1","#a5b4fc"],
                      text="Count")
        fig2.update_layout(**CHART_THEME, height=300, margin=dict(t=10,b=10,l=10,r=10),
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="Candidates")
        fig2.update_traces(textposition="outside", textfont_size=11)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Experience + CGPA histograms
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.markdown("<div class='section-header'>📅 Experience Distribution</div>", unsafe_allow_html=True)
        exp_clean = df["Experience_Years"].dropna()
        fig3 = px.histogram(exp_clean, nbins=30, color_discrete_sequence=["#6366f1"])
        fig3.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=10,l=10,r=10),
                           bargap=0.05, xaxis_title="Experience (Years)", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>🎓 CGPA Distribution</div>", unsafe_allow_html=True)
        cgpa_clean = df["CGPA"].dropna()
        fig4 = px.histogram(cgpa_clean, nbins=30, color_discrete_sequence=["#8b5cf6"])
        fig4.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=10,l=10,r=10),
                           bargap=0.05, xaxis_title="CGPA", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Education + Missing values
    col5, col6 = st.columns(2, gap="medium")
    with col5:
        st.markdown("<div class='section-header'>🎓 Education Breakdown</div>", unsafe_allow_html=True)
        edu_counts = df["Education"].value_counts().reset_index()
        edu_counts.columns = ["Education", "Count"]
        fig5 = px.bar(edu_counts, x="Education", y="Count",
                      color="Count", color_continuous_scale=["#1e1b4b","#6366f1","#c7d2fe"],
                      text="Count")
        fig5.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=10,l=10,r=10),
                           coloraxis_showscale=False, xaxis_title="", yaxis_title="Candidates")
        fig5.update_traces(textposition="outside")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("<div class='section-header'>🔴 Missing Values by Column</div>", unsafe_allow_html=True)
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=True)
        if len(miss) > 0:
            fig6 = px.bar(miss, x="Missing", y="Column", orientation="h",
                          color="Missing", color_continuous_scale=["#7f1d1d","#ef4444","#fca5a5"],
                          text="Missing")
            fig6.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=10,l=10,r=10),
                               coloraxis_showscale=False, yaxis_title="", xaxis_title="Missing Count")
            fig6.update_traces(textposition="outside")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Shortlist rate by role + Scatter
    col7, col8 = st.columns(2, gap="medium")
    with col7:
        st.markdown("<div class='section-header'>🏆 Shortlist Rate by Job Role</div>", unsafe_allow_html=True)
        role_sl = df.groupby("Job_Role")["Shortlisted"].mean().mul(100).round(1).reset_index()
        role_sl.columns = ["Job_Role", "Shortlist_%"]
        role_sl = role_sl.sort_values("Shortlist_%")
        fig7 = px.bar(role_sl, x="Shortlist_%", y="Job_Role", orientation="h",
                      color="Shortlist_%", color_continuous_scale=["#1e3a5f","#06b6d4","#a5f3fc"],
                      text=role_sl["Shortlist_%"].apply(lambda x: f"{x}%"))
        fig7.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=10,l=10,r=10),
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="Shortlist Rate (%)")
        fig7.update_traces(textposition="outside")
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        st.markdown("<div class='section-header'>📊 Experience vs CGPA</div>", unsafe_allow_html=True)
        sample = df.dropna(subset=["Experience_Years","CGPA"]).sample(min(300, len(df)), random_state=42)
        sample = sample.copy()
        sample["Status"] = sample["Shortlisted"].map({1:"Shortlisted", 0:"Rejected"})
        fig8 = px.scatter(sample, x="Experience_Years", y="CGPA", color="Status",
                          color_discrete_map={"Shortlisted":"#4ade80","Rejected":"#f87171"},
                          opacity=0.65)
        fig8.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=10,l=10,r=10),
                           legend=dict(orientation="h", y=1.08),
                           xaxis_title="Experience (Years)", yaxis_title="CGPA")
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Numerical summary
    st.markdown("<div class='section-header'>📐 Numerical Summary Statistics</div>", unsafe_allow_html=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    desc = df[num_cols].describe().T.round(2).reset_index().rename(columns={"index": "Feature"})
    st.dataframe(desc, use_container_width=True)

    # Day 1 banner
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-banner'>
        <strong>✅ Day 1 Complete — Dataset Exploration</strong><br>
        Loaded <strong>{total_rows:,} resumes</strong> with <strong>{total_cols} features</strong>.
        Found <strong>{missing_vals} missing values</strong> ({missing_pct}%) and
        <strong>{duplicates} duplicate rows</strong>. Acceptance rate: <strong>{shortlist_pct}%</strong>.<br>
        Key insight: Experience ≥ 2.5 yrs and CGPA ≥ 7.5 are the strongest shortlisting signals.<br><br>
        <strong>Next →</strong> Week 2: EDA &amp; Insights (correlation heatmap, outlier detection, skill frequency analysis)
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Other pages
# ═══════════════════════════════════════════════════════════════════════════════
else:
    page_map = {
        "EDA":                ("🔍", "EDA & Insights",             "Week 2"),
        "Data Cleaning":      ("🧹", "Data Cleaning",              "Week 3"),
        "Feature Engineering":("⚙️",  "Feature Engineering",        "Week 4"),
        "Model Training":     ("🤖", "Model Training",             "Week 5"),
        "Model Evaluation":   ("📈", "Model Evaluation",           "Week 6"),
        "Full Pipeline":      ("🚀", "Full ML Pipeline",           "Week 7"),
        "Resume Ranker":      ("🗂️",  "Live Resume Ranker",         "Week 8"),
    }
    for key, (icon, title, week) in page_map.items():
        if key in menu:
            st.markdown(f"""
            <div class='top-header'>
                <span style='font-size:1.6rem;'>{icon}</span>
                <b style='font-size:1.3rem; color:#e2e8f0;'> {title}</b>
                <div style='font-size:0.8rem; color:#64748b; margin-top:4px;'>{week} — Coming soon</div>
            </div>
            <div class='info-banner'>🚧 This module will be built in {week}. Stay tuned!</div>
            """, unsafe_allow_html=True)
            break
