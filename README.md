# AI-Based Resume Screening and Candidate Shortlisting System

A 9-week machine learning project for automated resume evaluation and candidate shortlisting, developed as part of a final-year B.Tech project under the guidance of HCL Tech.

---

## 1. Project Overview

Recruitment teams spend substantial time manually screening resumes for technical roles. This project develops an end-to-end machine learning pipeline to assist technical recruiters by ranking candidates and predicting shortlisting outcomes based on skills, experience, academic credentials, and project history.

### Core Objectives
- Automate first-round candidate filtering using supervised machine learning and natural language processing.
- Provide objective, reproducible evaluation scores for applicants across multiple technical domains.
- Identify candidate skill gaps and provide actionable recommendations for profile enhancement.
- Deploy an intuitive web interface for technical recruiters and hiring managers.

---

## 2. Repository Structure

```text
resume-screening/
|-- data/
|   |-- raw_resumes.csv                 # Raw dataset (1,020 candidate profiles)
|   `-- cleaned_resumes.csv             # Cleaned dataset (Week 3)
|-- models/                             # Serialized ML models and preprocessors (Week 5+)
|-- notebooks/
|   `-- week1_data_collection_eda.ipynb # Week 1 data collection and EDA notebook
|-- src/
|   `-- generate_data.py                # Synthetic dataset generation script
|-- app.py                              # Streamlit recruiter dashboard
|-- problem_statement.md                # Project problem definition and scope
|-- PROJECT_CONTEXT.md                  # Project milestones and working constraints
|-- README.md                           # Project documentation
`-- requirements.txt                    # Project dependencies
```

---

## 3. Weekly Milestones

| Week | Milestone | Status | Deliverables |
|---|---|---|---|
| Week 1 | Data Collection & EDA | Completed | `src/generate_data.py`, `data/raw_resumes.csv`, `notebooks/week1_data_collection_eda.ipynb` |
| Week 2 | EDA Viewer & Deep Dive | In Progress | Outlier analysis, correlation study, distribution analysis |
| Week 3 | Data Cleaning & Preprocessing | Planned | Duplicate removal, null imputation, text cleaning |
| Week 4 | Feature Engineering | Planned | TF-IDF extraction, skill encoding, similarity scoring |
| Week 5 | Baseline Model Training | Planned | Logistic Regression, Decision Tree, Random Forest |
| Week 6 | Hyperparameter Tuning | Planned | Grid search, cross-validation, model optimization |
| Week 7 | Model Evaluation | Planned | Confusion matrix, ROC-AUC, classification metrics |
| Week 8 | End-to-End Pipeline Runner | Planned | Scikit-learn Pipeline with ColumnTransformer |
| Week 9 | Final Recruiter Dashboard | Planned | Resume ranking, skill gap advisor, and score enhancement |

---

## 4. Week 1 Implementation Details

### Data Collection
- Generated a controlled dataset of 1,020 resumes simulating five major technical roles: Data Scientist, Machine Learning Engineer, Python Developer, Web Developer, and Data Analyst.
- Simulated real-world recruitment criteria with controlled data imperfections (missing values, duplicate submissions, and outlier values) to test data preprocessing pipelines.

### Dataset Schema
- **Candidate_ID**: Unique identifier for each candidate.
- **Resume_Text**: Textual overview including degree, background, and summary.
- **Skills**: Comma-separated list of candidate technical skills.
- **Experience_Years**: Total industry experience in years.
- **Education**: Highest qualification level (B.Tech, M.Tech, BCA, MCA, B.Sc, M.Sc).
- **CGPA**: Academic performance on a 10.0 scale.
- **Projects_Count**: Total technical projects completed.
- **Job_Role**: Target domain profile.
- **Shortlisted**: Binary outcome indicator (1 = Shortlisted, 0 = Rejected).

### Key Data Insights
- **Total Records**: 1,020
- **Shortlist Acceptance Rate**: ~65.9%
- **Missing Records**: 121 values (1.32% across Skills, Experience, Education, and CGPA)
- **Duplicate Records**: 20 rows
- **Key Determinants**: Shortlisted candidates show higher average experience (5.85 years vs 4.40 years) and higher average CGPA (8.18 vs 7.24).

---

## 5. Setup and Execution

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd resume-screening
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate the raw dataset (if not already generated):
   ```bash
   python src/generate_data.py
   ```

4. Launch the recruiter dashboard:
   ```bash
   python -m streamlit run app.py
   ```

5. Access the application in your browser at `http://localhost:8501`.

---

## 6. Technology Stack

- **Programming Language**: Python 3.10
- **Data Analysis & Visualization**: Pandas, NumPy, Plotly
- **Machine Learning**: Scikit-Learn
- **Web Interface**: Streamlit
- **Development Environment**: Jupyter Notebook, VS Code
