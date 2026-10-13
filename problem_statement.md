# Problem Statement

## Technical
Manual resume screening is a time-consuming, error-prone, and subjective process. HR teams spend 40–60% of their recruitment time shortlisting candidates from hundreds of resumes, leading to delays, bias, and missed talent. This project builds an AI-powered resume screening system that automatically analyzes candidate profiles and predicts shortlisting outcomes using machine learning.

## Why It Matters
- **Time cost:** Reviewing 500 resumes manually takes ~20 hours of HR effort
- **Bias:** Human reviewers unconsciously favor certain education brands, names, or writing styles
- **Inconsistency:** Different reviewers apply different standards for the same role
- **Scale:** High-volume hiring (100+ openings) is nearly impossible without automation

## Who Benefits
| Stakeholder | Benefit |
|---|---|
| HR Teams | Reduce screening time by 70–80% |
| Candidates | Faster, fairer shortlisting decisions |
| Companies | Better quality hires, lower cost-per-hire |
| Hiring Managers | Pre-filtered, ranked candidate pool |

## Solution Approach
Build a supervised ML pipeline that:
1. Ingests raw resume data (text + structured fields)
2. Performs EDA to understand patterns
3. Cleans and engineers features
4. Trains classification models to predict `Shortlisted` (0/1)
5. Evaluates models with precision, recall, F1-score
6. Deploys as an interactive Streamlit dashboard

## Dataset
- **Source:** Synthetically generated (realistic simulation of real resumes)
- **Size:** ~1,020 resumes (1,000 base + ~2% duplicates)
- **Features:** Candidate ID, Resume Text, Skills, Experience Years, Education, CGPA, Projects Count, Job Role
- **Target:** `Shortlisted` (1 = shortlisted, 0 = rejected)
- **Challenges:** Missing values (~3%), duplicates (~2%), outliers (8 injected)

## Success Metrics
- **Primary:** F1-Score ≥ 0.80 on test set
- **Secondary:** Precision ≥ 0.78, Recall ≥ 0.75
- **Fairness:** Shortlist rate should not vary by >5% across job roles for same qualifications
