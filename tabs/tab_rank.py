import streamlit as st
import pandas as pd
from src.match_scorer import compute_candidate_match, ROLE_SPECS

def render(df):
    st.subheader("Rank candidates")
    st.write("Select a job role to view ranked candidates.")

    if df is None or len(df) == 0:
        st.info("No candidates available to display.")
        return

    # role selection and filters
    all_roles = list(ROLE_SPECS.keys())
    if "Job_Role" in df.columns:
        dataset_roles = [r for r in df["Job_Role"].dropna().unique() if str(r).strip() in ROLE_SPECS]
        if dataset_roles:
            all_roles = sorted(dataset_roles)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel_role = st.selectbox("Job role", all_roles, index=0)
    with c2:
        top_n = st.slider("Show candidates", min_value=5, max_value=50, value=10, step=5)
    with c3:
        min_score = st.slider("Minimum match score", min_value=0, max_value=100, value=0, step=5)

    sub_df = df.copy()

    # compute score for each candidate against selected role
    match_results = []
    for _, row in sub_df.iterrows():
        res = compute_candidate_match(row, sel_role)
        match_results.append(res)

    res_df = pd.DataFrame(match_results)
    for col_name in res_df.columns:
        sub_df[col_name] = res_df[col_name].values

    # sort by final_score descending
    sub_df = sub_df.sort_values(by="final_score", ascending=False).reset_index(drop=True)

    # filter by minimum score
    sub_df = sub_df[sub_df["final_score"] * 100 >= min_score]

    if len(sub_df) == 0:
        st.info(f"No candidates found with score >= {min_score}% for {sel_role}.")
        return

    # prepare display rows
    records = []
    for rank_idx, (_, row) in enumerate(sub_df.head(top_n).iterrows(), start=1):
        score_val = float(row["final_score"])
        pct_score = int(round(score_val * 100))
        verdict = row["result_label"]

        records.append({
            "Rank": rank_idx,
            "Candidate ID": str(row.get("Candidate_ID", f"CID_{rank_idx}")),
            "Degree": str(row.get("Education", "B.Tech")),
            "Experience (years)": float(row.get("Experience_Years", 0.0)),
            "CGPA": float(row.get("CGPA", 0.0)),
            "Match score": f"{pct_score}%",
            "Result": verdict,
            "Reason": row["why"]
        })

    display_df = pd.DataFrame(records)

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )

    # download csv button
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download candidates (CSV)",
        data=csv_data,
        file_name=f"ranked_candidates_{sel_role.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

    # score explanation
    with st.expander("How score is calculated"):
        st.write("- Model probability: 60%")
        st.write("- Skill match: 30%")
        st.write("- Text similarity: 10%")
        st.write("- Candidates with degree mismatch or low skill match are ranked lower.")
