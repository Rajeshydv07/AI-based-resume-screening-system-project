import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# page config
st.set_page_config(
    page_title="AI Resume Screening Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# global css
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

# data loader
@st.cache_data
def load_data():
    path = os.path.join("data", "raw_resumes.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# sidebar
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
            "📊  W1: Dataset Preview",
            "🔍  W2: EDA Viewer",
            "🧹  W3: Cleaning Comparison",
            "⚙️  W4: Feature Engineering",
            "🤖  W5: Baseline Training",
            "🎯  W6: Model Tuning",
            "📈  W7: Evaluation",
            "🚀  W8: Pipeline Runner",
            "🗂️  W9: Dashboard + Extras",
        ],
        label_visibility="collapsed"
    )

    # W2 sidebar controls
    outlier_method = "IQR (k = 1.5)"
    target_col = "Experience_Years"
    filter_role = "All"
    filter_edu = "All"
    view_order = "Default"
    row_limit = 50

    if "W2" in menu or "EDA" in menu:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; font-weight:600; color:#6366f1; letter-spacing:.08em; margin-bottom:8px;'>⚙️ W2 CONTROLS</div>", unsafe_allow_html=True)
        outlier_method = st.selectbox("Outlier Detection Method", ["IQR (k = 1.5)", "IQR (k = 3.0)", "Z-Score (|z| > 3)"])
        target_col = st.selectbox("Numeric Column", ["Experience_Years", "CGPA", "Projects_Count"])
        df_temp = load_data()
        role_opts = ["All"] + sorted(df_temp["Job_Role"].dropna().unique().tolist()) if df_temp is not None else ["All"]
        edu_opts = ["All"] + sorted(df_temp["Education"].dropna().unique().tolist()) if df_temp is not None else ["All"]
        filter_role = st.selectbox("Filter Job Role", role_opts)
        filter_edu = st.selectbox("Filter Education", edu_opts)
        view_order = st.radio("Rows Order", ["Default", "Top N", "Bottom N"], horizontal=True)
        max_rows = len(df_temp) if df_temp is not None else 100
        row_limit = st.slider("Row Limit (N)", 10, max_rows, min(50, max_rows), 10)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; font-weight:600; color:#6366f1; letter-spacing:.08em; margin-bottom:8px;'>🗓️ PROJECT ROADMAP</div>", unsafe_allow_html=True)

    weeks = [
        ("Week 1", "Dataset Preview", "✅"),
        ("Week 2", "EDA Viewer", "✅"),
        ("Week 3", "Cleaning Comparison", "🔄"),
        ("Week 4", "Feature Engineering", "⏳"),
        ("Week 5", "Baseline Training", "⏳"),
        ("Week 6", "Model Tuning", "⏳"),
        ("Week 7", "Evaluation", "⏳"),
        ("Week 8", "Pipeline Runner", "⏳"),
        ("Week 9", "Dashboard + Extras", "⏳"),
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

# load data
df = load_data()

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font_color="#cbd5e1",
    title_font_color="#e2e8f0",
    colorway=["#6366f1","#8b5cf6","#ec4899","#06b6d4","#10b981","#f59e0b"],
)

# dataset preview
if "Dataset Explorer" in menu or "W1" in menu or "Dataset Preview" in menu:
    st.markdown("""
    <div class='top-header'>
        <span style='font-size:1.6rem;'>📊</span>
        <b style='font-size:1.3rem; color:#e2e8f0;'> Dataset Explorer</b>
        <div style='font-size:0.8rem; color:#64748b; margin-top:4px;'>Week 1 — Raw data inspection, schema analysis &amp; quality summary</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("⚠️ Dataset not found. Run `python src/dataset_generator.py` first.")
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

# week 2 eda viewer
elif "EDA" in menu or "W2" in menu:
    st.markdown("""
    <div class='top-header'>
        <span style='font-size:1.6rem;'>🔍</span>
        <b style='font-size:1.3rem; color:#e2e8f0;'> EDA Viewer</b>
        <div style='font-size:0.8rem; color:#64748b; margin-top:4px;'>Week 2 — Missing value report, outlier detection (IQR &amp; Z-score) &amp; distribution charts</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found. Run `python src/dataset_generator.py` first.")
        st.stop()

    # Outlier detection helper functions
    def find_outliers_iqr(df_in, col, k):
        s = df_in[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - (k * iqr)
        high = q3 + (k * iqr)
        mask = (df_in[col] < low) | (df_in[col] > high)
        return mask, low, high

    def find_outliers_zscore(df_in, col, thresh=3.0):
        s = df_in[col].dropna()
        mean = s.mean()
        std = s.std()
        if std == 0:
            return pd.Series(False, index=df_in.index), mean, mean
        z = (df_in[col] - mean).abs() / std
        mask = z > thresh
        low = mean - (thresh * std)
        high = mean + (thresh * std)
        return mask, low, high

    # Apply global filters
    df_filtered = df.copy()
    if filter_role != "All":
        df_filtered = df_filtered[df_filtered["Job_Role"] == filter_role]
    if filter_edu != "All":
        df_filtered = df_filtered[df_filtered["Education"] == filter_edu]
    if view_order == "Top N":
        df_filtered = df_filtered.head(row_limit)
    elif view_order == "Bottom N":
        df_filtered = df_filtered.tail(row_limit)

    # 5 Dynamic KPIs
    tot_res = len(df_filtered)
    short_ct = int(df_filtered["Shortlisted"].sum()) if ("Shortlisted" in df_filtered.columns and tot_res > 0) else 0
    short_pct = round((short_ct / tot_res) * 100, 1) if tot_res > 0 else 0.0
    avg_exp = round(df_filtered["Experience_Years"].mean(), 1) if (tot_res > 0 and "Experience_Years" in df_filtered.columns) else 0.0
    avg_cgpa = round(df_filtered["CGPA"].mean(), 2) if (tot_res > 0 and "CGPA" in df_filtered.columns) else 0.0
    total_cells = tot_res * len(df_filtered.columns) if tot_res > 0 else 1
    miss_pct = round((df_filtered.isnull().sum().sum() / total_cells) * 100, 1) if tot_res > 0 else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    for col_w, val, lbl, sub in [
        (k1, str(tot_res), "Total Resumes", "filtered count"),
        (k2, f"{short_pct}%", "Shortlisted", f"{short_ct} candidates"),
        (k3, f"{avg_exp} yrs", "Avg Experience", "in years"),
        (k4, str(avg_cgpa), "Avg CGPA", "out of 10"),
        (k5, f"{miss_pct}%", "Missing Rate", "of all cells"),
    ]:
        col_w.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{val}</div>
            <div class='lbl'>{lbl}</div>
            <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs: Missing Values | Outliers | Charts
    tab_miss, tab_out, tab_ch = st.tabs(["📋 Missing Values", "⚠️ Outliers", "📊 Charts"])

    # tab 1: missing values
    with tab_miss:
        st.markdown("<div class='section-header'>Missing Value Report</div>", unsafe_allow_html=True)

        miss_list = []
        for c in df_filtered.columns:
            m_count = int(df_filtered[c].isnull().sum())
            m_pct = round((m_count / tot_res) * 100, 2) if tot_res > 0 else 0.0
            miss_list.append({
                "Column": c,
                "Missing Count": m_count,
                "Missing %": m_pct,
                "Non-Null Count": tot_res - m_count,
                "Data Type": str(df_filtered[c].dtype)
            })
        miss_df = pd.DataFrame(miss_list)

        left_m, right_m = st.columns([3, 2], gap="medium")
        with left_m:
            st.dataframe(miss_df, use_container_width=True, height=300)
            csv_bytes = miss_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Missing Values CSV",
                data=csv_bytes,
                file_name="missing_values_report.csv",
                mime="text/csv"
            )
        with right_m:
            miss_bar = miss_df[miss_df["Missing Count"] > 0]
            if len(miss_bar) > 0:
                fig_miss = px.bar(
                    miss_bar,
                    x="Missing Count",
                    y="Column",
                    orientation="h",
                    color="Missing %",
                    color_continuous_scale=["#fca5a5", "#ef4444", "#7f1d1d"],
                    text="Missing Count"
                )
                fig_miss.update_layout(**CHART_THEME, height=300, margin=dict(t=10,b=10,l=10,r=10),
                                       coloraxis_showscale=False, yaxis_title="", xaxis_title="Missing Count")
                fig_miss.update_traces(textposition="outside")
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("No missing values in the filtered dataset.")

    # tab 2: outliers
    with tab_out:
        st.markdown("<div class='section-header'>Outlier Detection Summary</div>", unsafe_allow_html=True)
        st.caption(f"Active Method: {outlier_method}")

        num_cols = ["Experience_Years", "CGPA", "Projects_Count"]
        outlier_rows = []
        outlier_data = {}

        for c in num_cols:
            if c in df_filtered.columns:
                if "1.5" in outlier_method:
                    mask, low, high = find_outliers_iqr(df_filtered, c, 1.5)
                elif "3.0" in outlier_method:
                    mask, low, high = find_outliers_iqr(df_filtered, c, 3.0)
                else:
                    mask, low, high = find_outliers_zscore(df_filtered, c, 3.0)

                outlier_data[c] = (mask, low, high)
                ct = int(mask.sum())
                pct = round((ct / tot_res) * 100, 2) if tot_res > 0 else 0.0
                outlier_rows.append({
                    "Feature": c,
                    "Outlier Count": ct,
                    "Outlier %": pct,
                    "Lower Bound": round(low, 2),
                    "Upper Bound": round(high, 2),
                    "Min in Data": round(df_filtered[c].min(), 2) if tot_res > 0 else 0,
                    "Max in Data": round(df_filtered[c].max(), 2) if tot_res > 0 else 0
                })

        outlier_summary_df = pd.DataFrame(outlier_rows)
        st.dataframe(outlier_summary_df, use_container_width=True)

        st.markdown(f"<div class='section-header'>Flagged Outlier Rows ({target_col})</div>", unsafe_allow_html=True)
        if target_col in outlier_data:
            mask_t, low_t, high_t = outlier_data[target_col]
            flagged_rows = df_filtered[mask_t]
            st.write(f"Detected **{len(flagged_rows)}** outlier rows outside `[{round(low_t,2)}, {round(high_t,2)}]`:")
            if len(flagged_rows) > 0:
                display_cols = [c for c in ["Candidate_ID", "Job_Role", "Education", target_col, "Shortlisted", "Skills"] if c in flagged_rows.columns]
                st.dataframe(flagged_rows[display_cols], use_container_width=True)
            else:
                st.info(f"No outliers found in {target_col} under current settings.")

    # tab 3: charts
    with tab_ch:
        df_chart = df_filtered.copy()
        if "Shortlisted" in df_chart.columns:
            df_chart["Status"] = df_chart["Shortlisted"].map({1: "Shortlisted", 0: "Rejected"}).fillna("Unknown")
        else:
            df_chart["Status"] = "Unknown"

        c_box, c_viol = st.columns(2, gap="medium")
        with c_box:
            st.markdown(f"<div class='section-header'>Box Plot ({target_col})</div>", unsafe_allow_html=True)
            fig_box = px.box(
                df_chart.dropna(subset=[target_col]),
                x="Job_Role",
                y=target_col,
                color="Status",
                color_discrete_map={"Shortlisted": "#4ade80", "Rejected": "#f87171"},
                points="outliers"
            )
            fig_box.update_layout(**CHART_THEME, height=330, margin=dict(t=10,b=10,l=10,r=10),
                                  xaxis_title="", yaxis_title=target_col,
                                  legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_box, use_container_width=True)

        with c_viol:
            st.markdown(f"<div class='section-header'>Violin Distribution ({target_col})</div>", unsafe_allow_html=True)
            fig_viol = px.violin(
                df_chart.dropna(subset=[target_col]),
                x="Status",
                y=target_col,
                color="Status",
                color_discrete_map={"Shortlisted": "#4ade80", "Rejected": "#f87171"},
                box=True,
                points="all"
            )
            fig_viol.update_layout(**CHART_THEME, height=330, margin=dict(t=10,b=10,l=10,r=10),
                                   xaxis_title="", yaxis_title=target_col,
                                   legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_viol, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        c_heat, c_scat = st.columns(2, gap="medium")
        with c_heat:
            st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
            heat_cols = [c for c in ["Experience_Years", "CGPA", "Projects_Count", "Shortlisted"] if c in df_filtered.columns]
            if len(heat_cols) > 1 and len(df_filtered) > 1:
                corr_matrix = df_filtered[heat_cols].corr().round(2)
                fig_heat = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    zmin=-1,
                    zmax=1
                )
                fig_heat.update_layout(**CHART_THEME, height=330, margin=dict(t=10,b=10,l=10,r=10))
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Insufficient data to compute correlation matrix.")

        with c_scat:
            st.markdown("<div class='section-header'>Experience vs CGPA Scatter</div>", unsafe_allow_html=True)
            if "Experience_Years" in df_chart.columns and "CGPA" in df_chart.columns:
                scat_sample = df_chart.dropna(subset=["Experience_Years", "CGPA"])
                fig_scat = px.scatter(
                    scat_sample,
                    x="Experience_Years",
                    y="CGPA",
                    color="Status",
                    color_discrete_map={"Shortlisted": "#4ade80", "Rejected": "#f87171"},
                    hover_data=["Candidate_ID", "Job_Role"],
                    opacity=0.75
                )
                fig_scat.update_layout(**CHART_THEME, height=330, margin=dict(t=10,b=10,l=10,r=10),
                                       xaxis_title="Experience (Years)", yaxis_title="CGPA",
                                       legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-banner'>
        <strong>Week 2 Complete — EDA Viewer</strong><br>
        Analyzed missing value patterns and detected outliers across numerical features using {outlier_method}.<br>
        Generated correlation matrix and distribution plots (box, violin, scatter).<br><br>
        <strong>Next &rarr;</strong> Week 3: Cleaning Comparison (duplicate removal, null imputation, capping outliers, text cleaning)
    </div>
    """, unsafe_allow_html=True)

# other pages
else:
    page_map = {
        "W3": ("🧹", "W3: Cleaning Comparison", "Week 3"),
        "W4": ("⚙️",  "W4: Feature Engineering",  "Week 4"),
        "W5": ("🤖", "W5: Baseline Training",   "Week 5"),
        "W6": ("🎯", "W6: Model Tuning",        "Week 6"),
        "W7": ("📈", "W7: Evaluation",          "Week 7"),
        "W8": ("🚀", "W8: Pipeline Runner",     "Week 8"),
        "W9": ("🗂️",  "W9: Dashboard + Extras",  "Week 9"),
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
