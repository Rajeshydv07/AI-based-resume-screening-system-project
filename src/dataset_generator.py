import os
import random
import numpy as np
import pandas as pd

# seed for reproducible data
np.random.seed(42)
random.seed(42)

roles = [
    'Software Engineer',
    'Data Scientist',
    'Python Developer',
    'Web Developer',
    'Machine Learning Engineer',
    'Data Analyst',
    'DevOps Engineer',
    'Cloud Engineer'
]

skills_dict = {
    'Software Engineer': ['Java', 'Python', 'Data Structures', 'Algorithms', 'SQL', 'Git'],
    'Data Scientist': ['Python', 'Machine Learning', 'SQL', 'Pandas', 'Statistics', 'NLP'],
    'Python Developer': ['Python', 'Django', 'Flask', 'PostgreSQL', 'Docker', 'Git'],
    'Web Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js', 'MongoDB'],
    'Machine Learning Engineer': ['Python', 'TensorFlow', 'PyTorch', 'MLOps', 'Docker', 'Deep Learning'],
    'Data Analyst': ['SQL', 'Excel', 'Tableau', 'Power BI', 'Python', 'Statistics'],
    'DevOps Engineer': ['Docker', 'Kubernetes', 'Linux', 'CI/CD', 'AWS', 'Git'],
    'Cloud Engineer': ['AWS', 'Azure', 'Linux', 'Python', 'Docker', 'Networking']
}

educations = ['B.Tech', 'M.Tech', 'B.Sc', 'M.Sc', 'BCA', 'MCA']

def make_dataset(n=1000):
    rows = []
    
    for i in range(n):
        cid = 'CID_' + str(1001 + i)
        role = random.choice(roles)
        edu = random.choice(educations)
        
        # realistic random skills
        pool = skills_dict[role]
        k = random.randint(2, len(pool))
        cand_skills = random.sample(pool, k)
        skills_str = ', '.join(cand_skills)
        
        exp = round(random.uniform(0.5, 10.0), 1)
        cgpa = round(random.uniform(6.0, 9.8), 2)
        proj = random.randint(1, 7)
        
        # realistic shortlisting criteria
        score = 0
        if exp >= 2.5:
            score += 2
        if cgpa >= 7.5:
            score += 2
        if proj >= 3:
            score += 1
        if edu in ['B.Tech', 'M.Tech']:
            score += 1
        if len(cand_skills) >= 4:
            score += 1
            
        # random chance based on score
        prob = score / 7.0
        shortlisted = 1 if (prob + random.uniform(-0.15, 0.15)) >= 0.55 else 0
        
        # resume text
        txt = (
            f"Passionate {role} graduate with {edu} degree. "
            f"Demonstrated track record of {exp} years working with {skills_str}. "
            f"Completed {proj} technical projects with strong problem solving abilities."
        )
        
        rows.append({
            'Candidate_ID': cid,
            'Resume_Text': txt,
            'Skills': skills_str,
            'Experience_Years': exp,
            'Education': edu,
            'CGPA': cgpa,
            'Projects_Count': proj,
            'Job_Role': role,
            'Shortlisted': shortlisted
        })
        
    df = pd.DataFrame(rows)
    
    # add a few outliers (about 0.5% - 1%)
    outlier_idx = np.random.choice(n, size=8, replace=False)
    df.loc[outlier_idx[0], 'Experience_Years'] = 28.0
    df.loc[outlier_idx[1], 'Experience_Years'] = 35.0
    df.loc[outlier_idx[2], 'CGPA'] = 14.5
    df.loc[outlier_idx[3], 'CGPA'] = 2.1
    df.loc[outlier_idx[4], 'Projects_Count'] = 25
    df.loc[outlier_idx[5], 'Projects_Count'] = 30
    df.loc[outlier_idx[6], 'Experience_Years'] = -2.0
    df.loc[outlier_idx[7], 'CGPA'] = 0.5
    
    # add missing values (about 3%)
    for col in ['Experience_Years', 'CGPA', 'Education', 'Skills']:
        null_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[null_idx, col] = np.nan
        
    # add duplicates (about 2%)
    dup_rows = df.sample(n=int(n * 0.02), random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

def main():
    # ensure data folder exists
    os.makedirs('data', exist_ok=True)
    df = make_dataset(1000)
    save_path = os.path.join('data', 'raw_resumes.csv')
    df.to_csv(save_path, index=False)
    print("Dataset saved to", save_path)
    print("Shape:", df.shape)

if __name__ == '__main__':
    main()
