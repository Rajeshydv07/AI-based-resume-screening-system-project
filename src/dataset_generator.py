import os
import random
import pandas as pd
import numpy as np

# fix random seed for reproducibility
random.seed(42)
np.random.seed(42)

# load role specifications from data/job_roles.csv
ROLES_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'job_roles.csv')
if os.path.exists(ROLES_FILE):
    roles_df = pd.read_csv(ROLES_FILE)
    roles = roles_df['Job_Role'].tolist()
    skills_dict = {}
    criteria_dict = {}
    for _, r in roles_df.iterrows():
        role_name = r['Job_Role'].strip()
        skills = [s.strip() for s in str(r['Required_Skills']).split(',') if s.strip()]
        skills_dict[role_name] = skills
        degrees = [d.strip() for d in str(r['Allowed_Degrees']).split(',') if d.strip()]
        criteria_dict[role_name] = {
            'min_cgpa': float(r['Min_CGPA']),
            'allowed_degrees': degrees,
            'min_exp': float(r['Min_Experience']),
            'skills': skills
        }
else:
    roles = [
        'Software Engineer', 'Data Scientist', 'Data Analyst',
        'Machine Learning Engineer', 'DevOps Engineer', 'Cloud Engineer',
        'Python Developer', 'Web Developer'
    ]
    skills_dict = {
        'Software Engineer': ['Python', 'Java', 'C++', 'Data Structures', 'Algorithms', 'Git', 'SQL', 'System Design', 'OOP'],
        'Data Scientist': ['Python', 'Machine Learning', 'Pandas', 'NumPy', 'Scikit-Learn', 'Statistics', 'Deep Learning', 'Data Analysis', 'SQL'],
        'Data Analyst': ['SQL', 'Excel', 'Power BI', 'Tableau', 'Python', 'Data Analysis', 'Statistics', 'Reporting', 'Dashboard'],
        'Machine Learning Engineer': ['Python', 'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow', 'Docker', 'MLOps', 'Scikit-Learn'],
        'DevOps Engineer': ['Docker', 'Kubernetes', 'AWS', 'Linux', 'CI/CD', 'Terraform', 'Bash', 'Git', 'Monitoring', 'Ansible'],
        'Cloud Engineer': ['AWS', 'Azure', 'Cloud Architecture', 'Kubernetes', 'Terraform', 'Networking', 'Security', 'Linux'],
        'Python Developer': ['Python', 'Django', 'Flask', 'SQL', 'REST APIs', 'Git', 'Docker', 'PostgreSQL'],
        'Web Developer': ['JavaScript', 'React', 'HTML', 'CSS', 'Node.js', 'REST APIs', 'Git', 'Express', 'MongoDB']
    }
    criteria_dict = {}

educations = ['B.Tech', 'M.Tech', 'B.Sc', 'M.Sc', 'BCA', 'MCA', 'Diploma']

# general other skills to simulate mixed or irrelevant profiles
other_skills = [
    'Photoshop', 'Graphic Design', 'Tally', 'Content Writing', 'Digital Marketing',
    'SEO', 'Accounting', 'Public Relations', 'AutoCAD', 'Video Editing', 'Salesforce'
]

def make_dataset(n=1000):
    rows = []
    
    for i in range(n):
        cid = 'CID_' + str(1001 + i)
        role = random.choice(roles)
        edu = random.choice(educations)
        
        pool = skills_dict[role]
        
        # simulate realistic candidate profiles:
        # some candidates have strong relevant skills, others have low/irrelevant skills
        profile_quality = random.choices(['strong', 'average', 'weak'], weights=[0.48, 0.32, 0.20])[0]
        
        if profile_quality == 'strong':
            k_relevant = random.randint(4, len(pool))
            k_other = random.randint(0, 1)
            exp = round(random.uniform(2.5, 9.5), 1)
            cgpa = round(random.uniform(7.5, 9.8), 2)
            proj = random.randint(3, 7)
        elif profile_quality == 'average':
            k_relevant = random.randint(2, 4)
            k_other = random.randint(1, 2)
            exp = round(random.uniform(1.2, 5.0), 1)
            cgpa = round(random.uniform(6.8, 8.4), 2)
            proj = random.randint(2, 4)
        else:
            k_relevant = random.randint(0, 2)
            k_other = random.randint(2, 4)
            exp = round(random.uniform(0.5, 3.0), 1)
            cgpa = round(random.uniform(5.5, 7.0), 2)
            proj = random.randint(1, 2)
            
        cand_relevant = random.sample(pool, min(k_relevant, len(pool)))
        cand_other = random.sample(other_skills, min(k_other, len(other_skills)))
        cand_skills = cand_relevant + cand_other
        random.shuffle(cand_skills)
        skills_str = ', '.join(cand_skills)
        
        # realistic, role-aware shortlisting criteria
        role_crit = criteria_dict.get(role, {})
        min_cgpa = role_crit.get('min_cgpa', 7.0)
        min_exp = role_crit.get('min_exp', 1.5)
        allowed_degs = role_crit.get('allowed_degrees', ['B.Tech', 'M.Tech', 'MCA'])
        
        skill_coverage = len(cand_relevant) / max(1, len(pool))
        deg_ok = edu in allowed_degs
        cgpa_ok = cgpa >= min_cgpa
        exp_ok = exp >= min_exp
        
        # hard qualification check
        if (not deg_ok) or (skill_coverage < 0.30) or (not cgpa_ok) or (not exp_ok):
            shortlisted = 0
        else:
            # qualified candidates
            prob = 0.45 * skill_coverage + 0.30 * (cgpa / 10.0) + 0.25 * min(1.0, exp / 5.0)
            shortlisted = 1 if prob >= 0.48 else 0
        
        txt = (
            f"Passionate {role} professional with {edu} degree. "
            f"Demonstrated background of {exp} years working with {skills_str}. "
            f"Completed {proj} major engineering projects with strong focus on performance and quality."
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
    
    # add realistic outlier records (about 0.5%)
    outlier_idx = np.random.choice(n, size=6, replace=False)
    df.loc[outlier_idx[0], 'Experience_Years'] = 28.0
    df.loc[outlier_idx[1], 'CGPA'] = 14.5
    df.loc[outlier_idx[2], 'Projects_Count'] = 25
    df.loc[outlier_idx[3], 'Experience_Years'] = -1.0
    
    # add missing values (about 3%)
    for col in ['Experience_Years', 'CGPA', 'Education', 'Skills']:
        null_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[null_idx, col] = np.nan
        
    # add duplicates (about 2%)
    dup_rows = df.sample(n=int(n * 0.02), random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

def main():
    os.makedirs('data', exist_ok=True)
    df = make_dataset(1000)
    save_path = os.path.join('data', 'raw_resumes.csv')
    df.to_csv(save_path, index=False)
    print("Role-aware raw dataset generated successfully!")
    print("Shape:", df.shape)
    print("Shortlisted rate:", round(df['Shortlisted'].mean() * 100, 2), "%")

if __name__ == '__main__':
    main()
