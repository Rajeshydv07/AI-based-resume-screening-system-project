import os
import sys
import pandas as pd
import numpy as np

# add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaner import clean_text, remove_duplicates, fill_missing, cap_outliers
from src.feature_engineer import add_skills_count, add_experience_bucket, EDU_MAP

def test_clean_text():
    sample = "Check https://myportfolio.com and email me at test@example.com! Built 5+ Python apps."
    cleaned = clean_text(sample)
    assert "https" not in cleaned
    assert "@" not in cleaned
    assert "python" in cleaned
    assert len(cleaned) > 0

def test_remove_duplicates():
    df = pd.DataFrame({
        'Candidate_ID': ['C1', 'C1', 'C2'],
        'CGPA': [8.5, 8.5, 9.0]
    })
    cleaned_df, dropped = remove_duplicates(df)
    assert dropped == 1
    assert len(cleaned_df) == 2

def test_add_skills_count():
    df = pd.DataFrame({
        'Skills': ['Python, SQL, React', 'Docker', np.nan, '']
    })
    df = add_skills_count(df)
    assert df.loc[0, 'Skills_Count'] == 3
    assert df.loc[1, 'Skills_Count'] == 1
    assert df.loc[2, 'Skills_Count'] == 0
    assert df.loc[3, 'Skills_Count'] == 0

def test_add_experience_bucket():
    df = pd.DataFrame({
        'Experience_Years': [1.0, 3.5, 7.0]
    })
    df = add_experience_bucket(df)
    assert df.loc[0, 'Experience_Bucket'] == 0  # Entry
    assert df.loc[1, 'Experience_Bucket'] == 1  # Mid
    assert df.loc[2, 'Experience_Bucket'] == 2  # Senior

if __name__ == '__main__':
    test_clean_text()
    test_remove_duplicates()
    test_add_skills_count()
    test_add_experience_bucket()
    print("All unit tests passed successfully!")
