import streamlit as st
import plotly.graph_objects as go
from src.match_scorer import ROLE_SPECS
from src.pdf_extractor import extract_pdf_text, extract_resume_fields

default_skills = "Python, SQL, Excel, Git"

def render():
    st.subheader("Improve a resume")
    st.write("Find missing skills and ways to improve the resume match score.")

    # file upload
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload resume to analyze (PDF or TXT)",
            type=["pdf", "txt"],
            help="Upload text-based PDF or text file",
            key="tab_improve_uploader"
        )

    # read uploaded file
    if uploaded_file is not None:
        file_key = f"improve_{uploaded_file.name}_{uploaded_file.size}"
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
                st.session_state["improve_resume_text"] = text
                st.session_state["improve_resume_fields"] = extract_resume_fields(text)
                st.session_state[file_key] = True
                st.success(f"File loaded: {uploaded_file.name}")

    fields = st.session_state.get("improve_resume_fields", {})

    with col2:
        detected_skills = fields.get("skills", "") if fields else ""
        skills_val = detected_skills if detected_skills and detected_skills != "Not found" else default_skills
        sample_skills = st.text_input("Candidate skills", skills_val, key="improve_skills")

    # detected details
    if fields:
        with st.expander("Detected details", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                st.text_input("Degree", value=fields.get("degree", "Not found"), key="improve_det_degree")
            with f2:
                st.text_input("CGPA", value=fields.get("cgpa", "Not found"), key="improve_det_cgpa")
            with f3:
                st.text_input("Experience (years)", value=fields.get("experience_years", "Not found"), key="improve_det_exp")
            st.write(f"**Skills found:** {fields.get('skills', 'Not found')}")

    # target role selection
    target_role = st.selectbox("Select target job role", list(ROLE_SPECS.keys()), key="improve_role")

    # calculate skill gap
    specs = ROLE_SPECS.get(target_role, list(ROLE_SPECS.values())[0])
    req_skills = specs['skills']

    cand_skills_list = [s.strip().lower() for s in sample_skills.split(",") if s.strip()]
    resume_text = st.session_state.get("improve_resume_text", "")
    text_words = [w.lower() for w in resume_text.split()] if resume_text else []

    matched_skills = [s for s in req_skills if any(s in c for c in cand_skills_list) or any(s in w for w in text_words)]
    missing_skills = [s for s in req_skills if s not in matched_skills]

    current_match_rate = len(matched_skills) / max(1, len(req_skills))
    potential_match_rate = min(1.0, (len(matched_skills) + min(3, len(missing_skills))) / max(1, len(req_skills)))
    gain_pct = int(round((potential_match_rate - current_match_rate) * 100))

    st.write("---")

    # chart and score impact
    c1, c2 = st.columns([3, 2])

    with c1:
        st.write("**Skills coverage:**")
        skill_names = [s.title() for s in req_skills]
        coverage = [1 if s in matched_skills else 0 for s in req_skills]
        colors = ["#185FA5" if c == 1 else "#CBD5E1" for c in coverage]

        fig = go.Figure(go.Bar(
            x=skill_names,
            y=[100 for _ in coverage],
            marker=dict(color=colors),
            text=["Present" if c == 1 else "Missing" for c in coverage],
            textposition="auto"
        ))
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=30, l=10, r=10),
            yaxis=dict(showticklabels=False, range=[0, 115]),
            xaxis=dict(tickangle=-25)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.write("**Score impact:**")
        st.metric("Current skill match", f"{int(round(current_match_rate * 100))}%")
        st.metric("Score after adding top skills", f"{int(round(potential_match_rate * 100))}%", delta=f"+{gain_pct}%")

    # missing skills and tips
    col_left, col_right = st.columns(2)

    with col_left:
        st.write("**Top skills to add:**")
        if missing_skills:
            for i, skill in enumerate(missing_skills[:5], start=1):
                st.write(f"{i}. **{skill}** (Required for {target_role})")
        else:
            st.success("All required skills are present in the resume.")

    with col_right:
        st.write("**Resume tips:**")
        st.write("- Mention project numbers and metrics (e.g. reduced load time by 15%).")
        st.write("- Add key technologies in project descriptions.")
        st.write("- Use standard technical terms like REST APIs, Docker, SQL.")
        st.write("- Add GitHub profile and certificates.")

    with st.expander("How score is calculated"):
        st.write("- Checks skills against role requirements.")
        st.write("- Estimates score boost after adding missing skills.")
