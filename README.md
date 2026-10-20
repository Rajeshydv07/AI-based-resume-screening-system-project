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
|   |-- cleaned_resumes.csv             # Sanitized dataset without duplicates/outliers
|   |-- X_train.csv / X_test.csv        # Engineered feature matrices (80/20 split)
|   `-- pipeline_predictions.csv        # Batch prediction outputs
|-- models/
|   |-- model.pkl                       # Tuned production champion model
|   |-- tfidf.pkl                       # Fitted TF-IDF vectorizer (max_features=500)
|   |-- minmax_scaler.pkl               # Fitted numerical scaler
|   `-- tuning_results.csv              # Hyperparameter search results
|-- notebooks/
|   `-- week1_data_collection_eda.ipynb # EDA notebook
|-- src/
|   |-- dataset_generator.py            # Synthetic dataset generator
|   |-- data_cleaner.py                 # Cleaning & text normalization module
|   |-- feature_engineer.py             # Feature extraction and encoding pipeline
|   |-- model_trainer.py                # 5-fold CV baseline model trainer
|   |-- model_tuner.py                  # GridSearchCV hyperparameter optimization
|   |-- model_evaluator.py              # Performance evaluation & ROC/PR curves
|   `-- pipeline.py                     # Production pipeline runner
|-- tests/
|   `-- test_pipeline.py                # Automated unit tests
|-- app.py                              # Streamlit recruiter dashboard
|-- problem_statement.md                # Project problem definition and scope
|-- PROJECT_CONTEXT.md                  # Project milestones and specifications
|-- README.md                           # Documentation
`-- requirements.txt                    # Python dependencies
```

---

## 3. Weekly Milestones

| Week | Milestone | Status | Deliverables |
|---|---|---|---|
| Week 1 | Data Collection & EDA | Completed | `src/dataset_generator.py`, `data/raw_resumes.csv`, `notebooks/week1_data_collection_eda.ipynb` |
| Week 2 | EDA Viewer & Deep Dive | Completed | `app.py` (W2: EDA Viewer), Outlier detection (IQR & Z-score), Missing report & CSV export, Plotly charts |
| Week 3 | Data Cleaning & Preprocessing | Completed | `src/data_cleaner.py`, `data/cleaned_resumes.csv`, `app.py` (W3: Cleaning Comparison) |
| Week 4 | Feature Engineering | Completed | `src/feature_engineer.py`, TF-IDF, ordinal/one-hot encoding, cosine match scoring |
| Week 5 | Baseline Model Training | Completed | `src/model_trainer.py`, Logistic Regression, Naive Bayes, Random Forest, SVM (5-fold CV) |
| Week 6 | Hyperparameter Tuning | Completed | `src/model_tuner.py`, GridSearchCV parameter optimization, champion `models/model.pkl` |
| Week 7 | Model Evaluation | Completed | `src/model_evaluator.py`, Confusion matrix, ROC curve (AUC 0.869), PR curve, feature importance |
| Week 8 | End-to-End Pipeline Runner | Completed | `src/pipeline.py`, unified class execution, `tests/test_pipeline.py` unit testing |
| Week 9 | Final Recruiter Dashboard | Completed | Job description ranking, instant single-resume analyzer, skill gap advisor |

---

## 4. Week 1 Implementation Details

### Data Collection
- Generated a controlled dataset of 1,020 resumes simulating eight technical roles: Software Engineer, Data Scientist, Python Developer, Web Developer, Machine Learning Engineer, Data Analyst, DevOps Engineer, and Cloud Engineer.
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
   python src/dataset_generator.py
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
