import os
import sys
import unittest
import pandas as pd
import numpy as np

# add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data_cleaner import clean_text, remove_duplicates, fill_missing, cap_outliers
from src.feature_engineer import add_skills_count, add_experience_bucket, EDU_MAP
from src.pdf_extractor import extract_resume_fields
from src.match_scorer import compute_candidate_match, ROLE_SPECS
from src.pipeline import ResumeScreeningPipeline

class TestResumeScreening(unittest.TestCase):

    def test_clean_text(self):
        sample = "Check https://myportfolio.com and email me at test@example.com! Built 5+ Python apps."
        cleaned = clean_text(sample)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertIn("python", cleaned)
        self.assertTrue(len(cleaned) > 0)

    def test_remove_duplicates(self):
        df = pd.DataFrame({
            'Candidate_ID': ['C1', 'C1', 'C2'],
            'CGPA': [8.5, 8.5, 9.0]
        })
        cleaned_df, dropped = remove_duplicates(df)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(cleaned_df), 2)

    def test_add_skills_count(self):
        df = pd.DataFrame({
            'Skills': ['Python, SQL, React', 'Docker', np.nan, '']
        })
        df = add_skills_count(df)
        self.assertEqual(df.loc[0, 'Skills_Count'], 3)
        self.assertEqual(df.loc[1, 'Skills_Count'], 1)
        self.assertEqual(df.loc[2, 'Skills_Count'], 0)
        self.assertEqual(df.loc[3, 'Skills_Count'], 0)

    def test_add_experience_bucket(self):
        df = pd.DataFrame({
            'Experience_Years': [1.0, 3.5, 7.0]
        })
        df = add_experience_bucket(df)
        self.assertEqual(df.loc[0, 'Experience_Bucket'], 0)  # Entry
        self.assertEqual(df.loc[1, 'Experience_Bucket'], 1)  # Mid
        self.assertEqual(df.loc[2, 'Experience_Bucket'], 2)  # Senior

    def test_pdf_extractor_fields(self):
        sample_resume_text = (
            "John Doe. B.Tech Computer Science graduate with CGPA 8.4/10. "
            "4+ years of experience as a software engineer with Python, SQL, Docker, and AWS."
        )
        fields = extract_resume_fields(sample_resume_text)
        self.assertEqual(fields['degree'], 'B.Tech')
        self.assertIn('python', fields['skills'].lower())
        self.assertIn('docker', fields['skills'].lower())
        self.assertEqual(fields['cgpa'], '8.4')
        self.assertEqual(fields['experience_years'], '4.0')

    def test_compute_candidate_match(self):
        candidate = {
            'Skills': 'Python, Java, Git, SQL, System Design, Data Structures, Algorithms',
            'Resume_Text': 'Built scalable backends in Python and SQL with OOP architecture.',
            'Education': 'B.Tech',
            'CGPA': 8.5,
            'Experience_Years': 3.5,
            'Shortlist_Probability': 0.85
        }
        res = compute_candidate_match(candidate, 'Software Engineer')
        self.assertIn(res['result_label'], ['Strong match', 'Possible match'])
        self.assertTrue(res['degree_ok'])
        self.assertTrue(res['cgpa_ok'])
        self.assertGreater(res['final_score'], 0.5)

    def test_pipeline_execution(self):
        models_path = os.path.join(parent_dir, 'models')
        if os.path.exists(os.path.join(models_path, 'model.pkl')):
            pipe = ResumeScreeningPipeline(models_dir=models_path)
            sample_df = pd.DataFrame([{
                'Candidate_ID': 'TEST_001',
                'Resume_Text': 'Experienced python software engineer specializing in backend systems.',
                'Skills': 'Python, SQL, Git, Linux',
                'Experience_Years': 3.0,
                'Education': 'B.Tech',
                'CGPA': 8.0,
                'Projects_Count': 3,
                'Job_Role': 'Software Engineer'
            }])
            result = pipe.process_and_predict(sample_df)
            self.assertIn('Shortlist_Probability', result.columns)
            self.assertIn('Predicted_Shortlist', result.columns)
            self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()
