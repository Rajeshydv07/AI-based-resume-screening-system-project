import os
import pickle
import pandas as pd
import numpy as np
from src.data_cleaner import clean_text
from src.feature_engineer import add_skills_count, add_experience_bucket, compute_match_scores, EDU_MAP, ROLE_KEYWORDS

class ResumeScreeningPipeline:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        with open(os.path.join(models_dir, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(models_dir, 'tfidf.pkl'), 'rb') as f:
            self.tfidf = pickle.load(f)
        with open(os.path.join(models_dir, 'minmax_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)

    def process_and_predict(self, raw_df):
        df = raw_df.copy()

        # 1. handle duplicates and missing values/outliers if raw input
        from src.data_cleaner import fill_missing, cap_outliers
        df = fill_missing(df)
        df, _ = cap_outliers(df)

        # 2. clean text
        if 'Resume_Text' in df.columns:
            df['Cleaned_Resume_Text'] = df['Resume_Text'].apply(clean_text)
        else:
            df['Cleaned_Resume_Text'] = ""

        # 3. domain features
        df = add_skills_count(df)
        df = add_experience_bucket(df)
        df['Skill_Match_Score'] = compute_match_scores(df.rename(columns={'Cleaned_Resume_Text': 'Resume_Text'}), self.tfidf)
        
        # 4. ordinal education
        df['Education_Encoded'] = df['Education'].map(lambda x: EDU_MAP.get(x, 2))
        
        # 5. one-hot job role
        job_roles = sorted(list(ROLE_KEYWORDS.keys()))
        for role in job_roles:
            col_name = f"Role_{role.replace(' ', '_')}"
            df[col_name] = (df['Job_Role'] == role).astype(int)
            
        # 6. scaling
        num_cols = ['Experience_Years', 'CGPA', 'Projects_Count', 'Skills_Count', 'Skill_Match_Score']
        scaled_vals = self.scaler.transform(df[num_cols])
        scaled_df = pd.DataFrame(scaled_vals, columns=[f"{c}_scaled" for c in num_cols], index=df.index)
        
        # 6. tfidf
        tfidf_mat = self.tfidf.transform(df['Cleaned_Resume_Text'].fillna(''))
        tfidf_cols = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=tfidf_cols, index=df.index)
        
        # 7. assemble matrix matching feature_names
        role_cols = [f"Role_{r.replace(' ', '_')}" for r in job_roles]
        non_text_df = pd.concat([
            scaled_df,
            df[['Education_Encoded', 'Experience_Bucket'] + role_cols]
        ], axis=1)
        
        full_features = pd.concat([non_text_df, tfidf_df], axis=1)
        
        # align columns
        for c in self.feature_names:
            if c not in full_features.columns:
                full_features[c] = 0.0
        X_mat = full_features[self.feature_names].values
        X_mat = np.clip(X_mat, 0, None)
        
        # predictions
        probs = self.model.predict_proba(X_mat)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X_mat)
        preds = (probs >= 0.5).astype(int)
        
        df['Shortlist_Probability'] = np.round(probs, 4)
        df['Predicted_Shortlist'] = preds
        df['Shortlist_Status'] = np.where(preds == 1, 'Shortlisted', 'Rejected')
        
        return df

def run_pipeline_file(input_csv, output_csv, models_dir='models'):
    df = pd.read_csv(input_csv)
    pipe = ResumeScreeningPipeline(models_dir)
    res_df = pipe.process_and_predict(df)
    res_df.to_csv(output_csv, index=False)
    return res_df

if __name__ == '__main__':
    in_path = os.path.join('data', 'raw_resumes.csv')
    out_path = os.path.join('data', 'pipeline_predictions.csv')
    if os.path.exists(in_path) and os.path.exists(os.path.join('models', 'model.pkl')):
        res = run_pipeline_file(in_path, out_path)
        print("Pipeline executed. Output rows:", len(res))
