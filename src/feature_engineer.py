import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# role skill mapping for similarity scoring
ROLE_KEYWORDS = {
    'Software Engineer': 'python java c++ data structures algorithms git sql system design oop',
    'Data Scientist': 'python machine learning pandas numpy scikit-learn statistics deep learning data analysis sql',
    'Data Analyst': 'sql excel power bi tableau python data analysis statistics reporting dashboard',
    'Machine Learning Engineer': 'python machine learning deep learning pytorch tensorflow docker mlops scikit-learn',
    'DevOps Engineer': 'docker kubernetes aws linux ci/cd terraform bash git monitoring ansible',
    'Cloud Engineer': 'aws azure cloud architecture kubernetes terraform networking security linux',
    'Python Developer': 'python django flask sql rest apis git docker postgresql',
    'Web Developer': 'javascript react html css node.js rest apis git express mongodb'
}

# education mapping
EDU_MAP = {
    'High School': 0,
    'Diploma': 1,
    'B.Sc': 2,
    'BCA': 2,
    'B.Tech': 2,
    'M.Sc': 3,
    'MCA': 3,
    'M.Tech': 3,
    'PhD': 4
}

# calculate skills count
def add_skills_count(df):
    def count_skills(val):
        if not isinstance(val, str) or not val.strip():
            return 0
        return len([s.strip() for s in val.split(',') if s.strip()])
    df['Skills_Count'] = df['Skills'].apply(count_skills)
    return df

# calculate experience bucket
def add_experience_bucket(df):
    def bucket_exp(exp):
        if exp <= 2:
            return 0  # Entry (0-2 yrs)
        elif exp <= 5:
            return 1  # Mid (3-5 yrs)
        else:
            return 2  # Senior (6+ yrs)
    df['Experience_Bucket'] = df['Experience_Years'].apply(bucket_exp)
    return df

# compute skill match score using cosine similarity
def compute_match_scores(df, tfidf_model):
    scores = []
    for _, row in df.iterrows():
        role = row.get('Job_Role', '')
        resume_txt = str(row.get('Resume_Text', ''))
        role_kw = ROLE_KEYWORDS.get(role, 'computer science technology engineering')
        
        # transform both using fitted tfidf
        vecs = tfidf_model.transform([resume_txt, role_kw])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        scores.append(round(float(sim), 4))
    return scores

# engineer features
def engineer_features(data_path, models_dir):
    df = pd.read_csv(data_path)
    
    # 1. new domain features
    df = add_skills_count(df)
    df = add_experience_bucket(df)
    
    # 2. train test split first to avoid leakage
    X = df.drop(columns=['Shortlisted', 'Candidate_ID'])
    y = df['Shortlisted']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X_train.copy().reset_index(drop=True)
    X_test = X_test.copy().reset_index(drop=True)
    y_train = y_train.copy().reset_index(drop=True)
    y_test = y_test.copy().reset_index(drop=True)
    
    # 3. TF-IDF on Resume_Text
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    train_text = X_train['Resume_Text'].fillna('')
    test_text = X_test['Resume_Text'].fillna('')
    
    train_tfidf_matrix = tfidf.fit_transform(train_text)
    test_tfidf_matrix = tfidf.transform(test_text)
    
    tfidf_feature_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    df_tfidf_train = pd.DataFrame(train_tfidf_matrix.toarray(), columns=tfidf_feature_names)
    df_tfidf_test = pd.DataFrame(test_tfidf_matrix.toarray(), columns=tfidf_feature_names)
    
    # 4. Skill match score
    X_train['Skill_Match_Score'] = compute_match_scores(X_train, tfidf)
    X_test['Skill_Match_Score'] = compute_match_scores(X_test, tfidf)
    
    # 5. Ordinal encoding for Education
    X_train['Education_Encoded'] = X_train['Education'].map(lambda x: EDU_MAP.get(x, 2))
    X_test['Education_Encoded'] = X_test['Education'].map(lambda x: EDU_MAP.get(x, 2))
    
    # 6. One-hot encoding for Job_Role
    job_roles = sorted(list(ROLE_KEYWORDS.keys()))
    for role in job_roles:
        col_name = f"Role_{role.replace(' ', '_')}"
        X_train[col_name] = (X_train['Job_Role'] == role).astype(int)
        X_test[col_name] = (X_test['Job_Role'] == role).astype(int)
        
    # 7. Numerical scaling (MinMaxScaler and StandardScaler for comparison)
    num_cols = ['Experience_Years', 'CGPA', 'Projects_Count', 'Skills_Count', 'Skill_Match_Score']
    
    minmax = MinMaxScaler()
    standard = StandardScaler()
    
    minmax.fit(X_train[num_cols])
    standard.fit(X_train[num_cols])
    
    scaled_train_num = pd.DataFrame(
        minmax.transform(X_train[num_cols]),
        columns=[f"{c}_scaled" for c in num_cols]
    )
    scaled_test_num = pd.DataFrame(
        minmax.transform(X_test[num_cols]),
        columns=[f"{c}_scaled" for c in num_cols]
    )
    
    # 8. assemble final feature matrices
    role_cols = [f"Role_{r.replace(' ', '_')}" for r in job_roles]
    non_text_train = pd.concat([
        scaled_train_num,
        X_train[['Education_Encoded', 'Experience_Bucket'] + role_cols]
    ], axis=1)
    
    non_text_test = pd.concat([
        scaled_test_num,
        X_test[['Education_Encoded', 'Experience_Bucket'] + role_cols]
    ], axis=1)
    
    final_X_train = pd.concat([non_text_train, df_tfidf_train], axis=1)
    final_X_test = pd.concat([non_text_test, df_tfidf_test], axis=1)
    
    # save models and artifacts
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'tfidf.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(models_dir, 'minmax_scaler.pkl'), 'wb') as f:
        pickle.dump(minmax, f)
    with open(os.path.join(models_dir, 'standard_scaler.pkl'), 'wb') as f:
        pickle.dump(standard, f)
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(list(final_X_train.columns), f)
        
    data_dir = os.path.dirname(data_path)
    final_X_train.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
    final_X_test.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)
    
    return final_X_train, final_X_test, y_train, y_test

def main():
    data_path = os.path.join('data', 'cleaned_resumes.csv')
    models_dir = 'models'
    X_train, X_test, y_train, y_test = engineer_features(data_path, models_dir)
    print("Feature engineering complete!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

if __name__ == '__main__':
    main()
