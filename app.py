import os
import streamlit as st
import pandas as pd
import numpy as np

from ui.styles import APP_CSS
from tabs import tab_rank, tab_check, tab_improve
from src.match_scorer import compute_candidate_match
from src.pdf_extractor import extract_pdf_text, extract_resume_fields

# page config
st.set_page_config(
    page_title="ResumeIQ - Candidate Screening Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# inject clean accessible styles
st.markdown(APP_CSS, unsafe_allow_html=True)

REQUIRED_COLUMNS = ['Resume_Text', 'Skills', 'Experience_Years', 'Education', 'CGPA', 'Job_Role']

def get_file_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0

@st.cache_data
def load_sample_dataset(mtime):
    path = os.path.join("data", "pipeline_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    clean_path = os.path.join("data", "cleaned_resumes.csv")
    if os.path.exists(clean_path):
        return pd.read_csv(clean_path)
    raw_path = os.path.join("data", "raw_resumes.csv")
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return None

# sidebar
with st.sidebar:
    st.markdown("<div style='font-size:1.3rem; font-weight:700; color:#185FA5;'>ResumeIQ</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.82rem; color:#64748B; margin-bottom:1rem;'>Candidate screening platform</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E2E8F0; margin: 10px 0;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.82rem; font-weight:600; color:#334155;'>How to use</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#475569; line-height:1.5; margin-top:6px;'>
    1. <strong>Rank candidates:</strong> Select a job role to view prioritized applicants.<br><br>
    2. <strong>Check one resume:</strong> Upload a candidate CV to get instant match ratings.<br><br>
    3. <strong>Improve a resume:</strong> Review missing competencies and recommended keywords.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E2E8F0; margin: 16px 0;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.82rem; font-weight:600; color:#334155;'>Model information</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:10px; margin-top:6px; font-size:0.8rem;'>
        <div><strong>Model:</strong> Tuned Logistic Regression</div>
        <div style='margin-top:4px;'><strong>Test accuracy:</strong> 81.5%</div>
        <div style='margin-top:4px;'><strong>ROC-AUC:</strong> 86.9%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:3rem;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#E2E8F0; margin: 12px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748B;'>
        Developed by Final-Year B.Tech Team<br>
        <strong>ABES Engineering College</strong>
    </div>
    """, unsafe_allow_html=True)

# top main header
st.markdown("""
<div class='app-header'>
    <div class='app-title'>ResumeIQ</div>
    <div class='app-subtitle'>Rank candidates for a job, check a single resume, and see how to improve it.</div>
</div>
""", unsafe_allow_html=True)

# 1. Front Upload & Dataset Resolution (CSV + PDF)
sample_mtime = get_file_mtime(os.path.join("data", "pipeline_predictions.csv"))
sample_df = load_sample_dataset(sample_mtime)

active_df = None
active_badge_html = ""

with st.container():
    up_c1, up_c2 = st.columns([3, 1], gap="medium")

    with up_c1:
        uploaded_files = st.file_uploader(
            "Upload resumes (CSV or PDF — multiple files allowed)",
            type=["csv", "pdf"],
            accept_multiple_files=True,
            help="Upload raw_resumes.csv or individual PDF resumes. Do not upload pipeline output files like pipeline_predictions.csv.",
            key="main_uploader"
        )

    with up_c2:
        use_sample = st.checkbox(
            "Don't have a file? Use the sample dataset",
            value=(not uploaded_files),
            key="use_sample_chk"
        )
        # sample CSV download
        if sample_df is not None:
            sample_download_cols = ['Candidate_ID', 'Resume_Text', 'Skills', 'Experience_Years', 'Education', 'CGPA', 'Job_Role']
            export_sample = sample_df[[c for c in sample_download_cols if c in sample_df.columns]].head(5)
            sample_bytes = export_sample.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download sample format",
                sample_bytes,
                "sample_resumes_format.csv",
                "text/csv",
                help="Download sample CSV template matching required schema"
            )

    # separate uploaded files by type
    csv_files = [f for f in (uploaded_files or []) if f.name.lower().endswith(".csv")]
    pdf_files = [f for f in (uploaded_files or []) if f.name.lower().endswith(".pdf")]

    csv_df = None
    pdf_df = None

    # --- process CSV files ---
    for csv_file in csv_files:
        try:
            raw_df = pd.read_csv(csv_file)

            # detect if user uploaded an output/prediction file by mistake
            if 'Shortlist_Probability' in raw_df.columns or 'Shortlist_Status' in raw_df.columns:
                st.error(
                    f"{csv_file.name} looks like a pipeline output file (it has prediction columns). "
                    "Please upload the original resume CSV (e.g. raw_resumes.csv) instead."
                )
                continue

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
            if missing_cols:
                st.error(
                    f"{csv_file.name} is missing required columns: {', '.join(missing_cols)}. "
                    "Your CSV must have: Resume_Text, Skills, Experience_Years, Education, CGPA, Job_Role."
                )
                continue

            cache_key = f"processed_{csv_file.name}_{len(raw_df)}"
            if cache_key in st.session_state:
                result = st.session_state[cache_key]
            else:
                with st.spinner(f"Processing {csv_file.name} through ML pipeline..."):
                    from src.pipeline import ResumeScreeningPipeline
                    pipe = ResumeScreeningPipeline()
                    result = pipe.process_and_predict(raw_df)
                    st.session_state[cache_key] = result
            csv_df = pd.concat([csv_df, result], ignore_index=True) if csv_df is not None else result
        except Exception as e:
            st.error(f"Could not read {csv_file.name}: {e}")

    # --- process PDF files ---
    pdf_rows = []
    for pdf_file in pdf_files:
        cache_key = f"pdf_{pdf_file.name}_{pdf_file.size}"
        if cache_key in st.session_state:
            row = st.session_state[cache_key]
        else:
            text, err = extract_pdf_text(pdf_file)
            if err:
                st.warning(f"{pdf_file.name}: {err}")
                continue
            fields = extract_resume_fields(text)
            try:
                exp_val = float(fields['experience_years'])
            except Exception:
                exp_val = 2.0
            try:
                cgpa_val = float(fields['cgpa'])
            except Exception:
                cgpa_val = 7.0
            degree_val = fields['degree'] if fields['degree'] != 'Not found' else 'B.Tech'
            skills_val = fields['skills'] if fields['skills'] != 'Not found' else ''
            row = {
                'Candidate_ID': pdf_file.name.replace('.pdf', ''),
                'Resume_Text': text,
                'Skills': skills_val,
                'Experience_Years': exp_val,
                'Education': degree_val,
                'CGPA': cgpa_val,
                'Projects_Count': 2,
                # No fixed Job_Role — Rank tab scores all candidates against the selected role
                'Job_Role': 'Software Engineer',
            }
            st.session_state[cache_key] = row
        pdf_rows.append(row)

    if pdf_rows:
        pdf_raw_df = pd.DataFrame(pdf_rows)
        cache_key_pdf = f"pdf_pipeline_{len(pdf_rows)}_{sum(f.size for f in pdf_files)}"
        if cache_key_pdf in st.session_state:
            pdf_df = st.session_state[cache_key_pdf]
        else:
            with st.spinner(f"Running ML pipeline on {len(pdf_rows)} PDF resume(s)..."):
                from src.pipeline import ResumeScreeningPipeline
                pipe = ResumeScreeningPipeline()
                pdf_df = pipe.process_and_predict(pdf_raw_df)
                st.session_state[cache_key_pdf] = pdf_df

    # --- combine ---
    parts = [p for p in [csv_df, pdf_df] if p is not None]
    if parts:
        active_df = pd.concat(parts, ignore_index=True)

        # build badge text
        badge_parts = []
        if csv_files:
            badge_parts.append(f"{len(csv_files)} CSV file{'s' if len(csv_files)>1 else ''} ({len(csv_df):,} resumes)" if csv_df is not None else "")
        if pdf_files:
            badge_parts.append(f"{len(pdf_files)} PDF{'s' if len(pdf_files)>1 else ''}")
        badge_label = " + ".join([b for b in badge_parts if b])
        active_badge_html = (
            f"<div style='background-color:#EFF6FF; border:1px solid #BFDBFE; color:#1E40AF; "
            f"padding:6px 12px; border-radius:6px; font-size:0.83rem; margin-bottom:12px;'>"
            f"Using your files: <strong>{badge_label}</strong> &mdash; {len(active_df):,} resumes total</div>"
        )
    elif use_sample and sample_df is not None:
        active_df = sample_df
        active_badge_html = (
            f"<div style='background-color:#F1F5F9; border:1px solid #CBD5E1; color:#475569; "
            f"padding:6px 12px; border-radius:6px; font-size:0.83rem; margin-bottom:12px;'>"
            f"Using sample dataset (<strong>{len(sample_df):,} resumes</strong>)</div>"
        )
    else:
        active_df = None

if active_badge_html:
    st.markdown(active_badge_html, unsafe_allow_html=True)

# 2. KPI Cards
tot_resumes = len(active_df) if active_df is not None else 0

if active_df is not None:
    # compute composite scores across active dataset
    strong_cnt = 0
    match_sum = 0
    for _, row in active_df.iterrows():
        role = row.get("Job_Role", "Software Engineer")
        eval_dict = compute_candidate_match(row, role)
        if eval_dict["result_label"] == "Strong match":
            strong_cnt += 1
        match_sum += eval_dict["final_score"]

    strong_pct = round((strong_cnt / max(1, tot_resumes)) * 100, 1) if tot_resumes else 0.0
    avg_match = round((match_sum / max(1, tot_resumes)) * 100, 1) if tot_resumes else 0.0
else:
    strong_pct = 0.0
    avg_match = 0.0

common_skill = "Python"
if active_df is not None and "Skills" in active_df.columns:
    try:
        all_skills = []
        for s_entry in active_df["Skills"].dropna():
            all_skills.extend([s.strip().title() for s in str(s_entry).split(",") if s.strip()])
        if all_skills:
            common_skill = pd.Series(all_skills).value_counts().index[0]
    except Exception:
        common_skill = "Python"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Total resumes</div>
        <div class='kpi-val'>{tot_resumes:,}</div>
        <div class='kpi-sub'>Evaluated candidate profiles</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Strong matches</div>
        <div class='kpi-val'>{strong_pct}%</div>
        <div class='kpi-sub'>Composite match score >= 70%</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Average skill match</div>
        <div class='kpi-val'>{avg_match}%</div>
        <div class='kpi-sub'>Across active candidate pool</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Most common skill</div>
        <div class='kpi-val'>{common_skill}</div>
        <div class='kpi-sub'>Highest frequency technical keyword</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# 3. Main Tabs
t_rank, t_check, t_improve = st.tabs([
    "Rank candidates",
    "Check one resume",
    "Improve a resume"
])

with t_rank:
    tab_rank.render(active_df)

with t_check:
    tab_check.render()

with t_improve:
    tab_improve.render()
