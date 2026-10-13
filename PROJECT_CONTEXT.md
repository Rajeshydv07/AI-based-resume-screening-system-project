# PROJECT: AI-Based Resume Screening and Candidate Shortlisting System

I am a final-year B.Tech IT student. This is a 9-week Streamlit ML project
done under HCL guidance. Problem type: binary classification.
Target column: Shortlisted (1 = shortlisted, 0 = rejected).
Dataset columns: Candidate_ID, Resume_Text, Skills, Experience_Years,
Education, CGPA, Projects_Count, Job_Role, Shortlisted.

## WORKING RULES (apply to every week)
1. Before creating, editing or deleting any file, installing a package or
   running a command, tell me what you will do and wait for my
   confirmation. If anything is unclear, ask me first.
2. Do only the week I name. Do not build future weeks early.
3. Work one file at a time and ask "Confirm to continue?" after each.
4. Code style: simple beginner-level student code, plain functions, short
   variable names, no type hints, short simple comments, no emojis, no
   long docstrings, no extra explanation at the end of files.
5. Do not touch anything outside resume-screening/.
6. Never overwrite my own edits (for example problem_statement.md)
   without asking.
7. To avoid data leakage: split train and test first, then fit scalers
   and TF-IDF on training data only.

## APP STRUCTURE
app.py is a Streamlit app. Sidebar radio with 9 tabs (W1 to W9).
Unfinished tabs show "Coming in Week N".
Folders: data/, notebooks/, src/, models/. Also requirements.txt, README.md.

## 9-WEEK PLAN

Week 1 - Dataset Preview
- Problem statement (Technical, Why it matters, Who benefits, Benefits)
- src/dataset_generator.py: 1000 rows with pandas, numpy, faker; 8 job roles;
  realistic Shortlisted logic; about 3% missing, 2% duplicates, a few
  outliers; saves data/raw_resumes.csv
- UI: CSV upload, 4 KPIs (rows, columns, missing, duplicates), head()
  with slider, dtypes table, Shortlisted bar chart

Week 2 - EDA Viewer
- Missing Value Report: table plus downloadable CSV
- Outlier detection: IQR (k = 1.5 / 3.0) and Z-score (|z| > 3); per-column
  outlier count and %
- Plotly charts: px.bar (missing), px.box per column, px.violin
  distribution, px.imshow correlation heatmap, px.scatter
  (Experience vs CGPA, colour = Shortlisted)
- UI: tabs Missing Values | Outliers | Charts; sidebar with column
  selector, detection method, top/bottom N toggles; KPI strip of 5 KPIs
  (Total Resumes, Shortlisted %, Avg Experience, Avg CGPA, Missing %)
  that respects global filters

Week 3 - Cleaning Comparison
- Remove duplicates; missing values under 5% delete or fill, over 5% fill
  with mean/median/mode (do not delete)
- Outliers: do not delete, use capping (lower bound / upper bound) or
  median
- Text cleaning: lowercase, remove URLs/emails/punctuation, stopwords,
  lemmatization
- Save data/cleaned_resumes.csv
- UI: tabs Before | After | Differences | Visual Compare; change log

Week 4 - Feature Engineering
- Encoding: Education ordinal, Job_Role one-hot
- Scaling: MinMaxScaler (0-1), StandardScaler for comparison
- TF-IDF on cleaned Resume_Text (max_features=500, ngram_range=(1,2))
- New features: Skills_Count, Skill_Match_Score (cosine similarity),
  Experience_Bucket
- Save scaler and TF-IDF as .pkl in models/
- UI: tabs Encoding | Scaling | Text Features | Final Matrix

Week 5 - Baseline Training
- Split 80/20, stratify=y, random_state=42
- Logistic Regression (baseline), Naive Bayes, Random Forest, SVM
- 5-fold CV; table of accuracy, F1, training time
- UI: tabs Setup | Train | Compare

Week 6 - Model Tuning
- GridSearchCV / RandomizedSearchCV on top 2 models
- LR: C. RF: n_estimators, max_depth, min_samples_split. SVM: C, kernel
- class_weight='balanced' if imbalanced
- Save best model to models/model.pkl
- UI: tabs Search Space | Results | Best Model; before vs after chart

Week 7 - Evaluation
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix, ROC curve, Precision-Recall curve
- Feature importance, error analysis (misclassified resumes)
- UI: tabs Metrics | Confusion | Curves | Importance | Errors; threshold
  slider 0.3 to 0.7

Week 8 - Pipeline Runner
- sklearn Pipeline + ColumnTransformer (clean, encode, scale, TF-IDF, model)
- Raw CSV upload, predictions, downloadable CSV, stage-wise progress, log
- 2-3 pytest tests for cleaning and feature functions

Week 9 - Dashboard + Extras
- KPI strip: Total Resumes, Shortlisted %, Avg Match Score, Top Skill,
  Model Accuracy
- Job description box: rank all resumes by match score
- Top-N ranked candidates table with probability and CSV download
- Single resume upload (PDF/TXT): prediction, score, matched vs missing
  skills
- Enhance Score feature: skill gap identification, missing keyword recommendations,
  and profile improvement score advisor
- Extras: README with screenshots, requirements.txt, GitHub repo,
  Streamlit Cloud deployment, final report/PPT

## CURRENT STATUS
(Update this line at the end of every week.)
Week 1: completed
Week 2: next
