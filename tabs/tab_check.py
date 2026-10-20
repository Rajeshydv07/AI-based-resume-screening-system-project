import streamlit as st
import pandas as pd
from src.match_scorer import compute_candidate_match, ROLE_SPECS
from src.pdf_extractor import extract_pdf_text, extract_resume_fields

# sample resume text
default_resume = (
    "Software engineer with 4 years of experience building scalable Python and SQL backends. "
    "Hands-on expertise with Docker, AWS cloud deployment, Git version control, and REST APIs. "
    "Strong understanding of data structures, algorithms, and system design principles."
)

def render():
    st.subheader("Check one resume")
    st.write("Upload or paste a resume to check match for a job role.")

    # file upload
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload resume (PDF or TXT)",
            type=["pdf", "txt"],
            help="Upload text-based PDF or text file",
            key="tab_check_uploader"
        )

    # read uploaded file
    if uploaded_file is not None:
        file_key = f"check_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            if uploaded_file.name.lower().endswith(".pdf"):
                text, err = extract_pdf_text(uploaded_file)
                if err:
                    st.warning(err)
                    text = ""
            else:
                try:
                    text = uploaded_file.read().decode("utf-8", errors="ignore").strip()
                    err = ""
                except Exception:
                    text, err = "", "Could not read file."
                if err:
                    st.warning(err)

            if text:
                st.session_state["check_resume_text"] = text
                st.session_state["check_resume_fields"] = extract_resume_fields(text)
                st.session_state[file_key] = True
                st.success(f"File loaded: {uploaded_file.name}")

    current_text = st.session_state.get("check_resume_text", default_resume)

    with col2:
        resume_text = st.text_area(
            "Resume text",
            value=current_text,
            height=130,
            key="check_resume_textarea"
        )
        st.session_state["check_resume_text"] = resume_text

    # detected details
    fields = st.session_state.get("check_resume_fields", {})

    if fields:
        with st.expander("Detected details", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                det_degree = st.text_input("Degree", value=fields.get("degree", "Not found"), key="check_det_degree")
            with f2:
                det_cgpa = st.text_input("CGPA", value=fields.get("cgpa", "Not found"), key="check_det_cgpa")
            with f3:
                det_exp = st.text_input("Experience (years)", value=fields.get("experience_years", "Not found"), key="check_det_exp")
            st.write(f"**Skills found:** {fields.get('skills', 'Not found')}")
    else:
        det_degree = "B.Tech"
        det_cgpa = "7.5"
        det_exp = "2.0"

    # candidate inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_role = st.selectbox("Target job role", list(ROLE_SPECS.keys()))
    with c2:
        try:
            exp_val = float(det_exp)
            exp_val = min(max(exp_val, 0.0), 15.0)
        except Exception:
            exp_val = 3.5
        cand_exp = st.slider("Experience (years)", 0.0, 15.0, exp_val, 0.5)
    with c3:
        degree_list = ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc", "M.Sc", "Diploma", "PhD"]
        idx = degree_list.index(det_degree) if det_degree in degree_list else 0
        cand_edu = st.selectbox("Degree", degree_list, index=idx)
    with c4:
        try:
            cgpa_val = float(det_cgpa)
            cgpa_val = min(max(cgpa_val, 5.0), 10.0)
        except Exception:
            cgpa_val = 7.5
        cand_cgpa = st.slider("CGPA", 5.0, 10.0, cgpa_val, 0.1)

    # skills input
    skills_detected = fields.get("skills", "") if fields else ""
    default_skills = skills_detected if skills_detected and skills_detected != "Not found" else "Python, SQL, Docker, AWS, Git, REST APIs, System Design"
    cand_skills_input = st.text_input("Candidate skills (comma-separated)", default_skills)

    # check button
    if st.button("Check resume", type="primary"):
        with st.spinner("Checking resume..."):
            try:
                from src.pipeline import ResumeScreeningPipeline
                pipe = ResumeScreeningPipeline()
                df_input = pd.DataFrame([{
                    "Candidate_ID": "RES_001",
                    "Resume_Text": resume_text,
                    "Skills": cand_skills_input,
                    "Experience_Years": cand_exp,
                    "Education": cand_edu,
                    "CGPA": cand_cgpa,
                    "Projects_Count": 3,
                    "Job_Role": target_role
                }])
                res_df = pipe.process_and_predict(df_input)
                row_data = res_df.iloc[0]
            except Exception:
                row_data = {
                    "Shortlist_Probability": 0.5,
                    "Skills": cand_skills_input,
                    "Resume_Text": resume_text,
                    "Education": cand_edu,
                    "CGPA": cand_cgpa,
                    "Experience_Years": cand_exp
                }

            res = compute_candidate_match(row_data, target_role)
            score_pct = int(round(res['final_score'] * 100))
            verdict = res['result_label']

            st.write("---")

            # result metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Result", verdict)
            with m2:
                st.metric("Match score", f"{score_pct}%")
            with m3:
                st.metric("Target role", target_role)

            st.info(f"**Assessment:** {res['why']}")

            # skills matched vs missing
            sk1, sk2 = st.columns(2)
            with sk1:
                st.write("**Matched skills:**")
                if res['matched_skills']:
                    st.success(", ".join(res['matched_skills']))
                else:
                    st.write("No direct skills matched.")

            with sk2:
                st.write("**Missing skills:**")
                if res['missing_skills']:
                    st.warning(", ".join(res['missing_skills']))
                else:
                    st.success("All required skills matched.")

            # criteria check
            st.write("**Criteria check:**")
            cr1, cr2, cr3 = st.columns(3)
            with cr1:
                st.write(f"Degree: {'Pass' if res['degree_ok'] else 'Needs review'} ({cand_edu})")
            with cr2:
                st.write(f"Experience: {'Pass' if res['exp_gap'] >= 0 else 'Below required'} ({cand_exp} yrs)")
            with cr3:
                st.write(f"CGPA: {'Pass' if res['cgpa_ok'] else 'Below required'} ({cand_cgpa})")

    # calculation info
    with st.expander("How score is calculated"):
        st.write("- Skills match: 30%")
        st.write("- Model probability: 60%")
        st.write("- Text similarity: 10%")
        st.write("- Hard rules: Degree mismatch or skill match < 30% marks candidate as Not a match.")
