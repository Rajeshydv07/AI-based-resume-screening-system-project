import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load or fallback job roles metadata
def get_role_specs():
    roles_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'job_roles.csv')
    if os.path.exists(roles_path):
        df_roles = pd.read_csv(roles_path)
        specs = {}
        for _, r in df_roles.iterrows():
            role_name = r['Job_Role'].strip()
            skills_list = [s.strip().lower() for s in str(r['Required_Skills']).split(',') if s.strip()]
            degrees = [d.strip() for d in str(r['Allowed_Degrees']).split(',') if d.strip()]
            specs[role_name] = {
                'skills': skills_list,
                'min_cgpa': float(r['Min_CGPA']),
                'degrees': degrees,
                'min_exp': float(r['Min_Experience']),
                'desc': str(r['Job_Description'])
            }
        return specs
    # fallback defaults
    return {
        "Software Engineer": {
            "skills": ["python", "java", "c++", "data structures", "algorithms", "git", "sql", "system design", "oop"],
            "min_cgpa": 7.0,
            "degrees": ["B.Tech", "M.Tech", "BCA", "MCA"],
            "min_exp": 1.5,
            "desc": "Build scalable software applications with data structures and SQL."
        },
        "Data Scientist": {
            "skills": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "statistics", "deep learning", "data analysis", "sql"],
            "min_cgpa": 7.5,
            "degrees": ["B.Tech", "M.Tech", "M.Sc", "PhD", "BCA", "MCA"],
            "min_exp": 2.0,
            "desc": "Design machine learning pipelines and statistical models."
        },
        "Cloud Engineer": {
            "skills": ["aws", "azure", "cloud architecture", "kubernetes", "terraform", "networking", "security", "linux"],
            "min_cgpa": 7.0,
            "degrees": ["B.Tech", "M.Tech", "BCA", "MCA"],
            "min_exp": 2.0,
            "desc": "Architect and maintain cloud infrastructure in AWS or Azure."
        },
        "Web Developer": {
            "skills": ["javascript", "react", "html", "css", "node.js", "rest apis", "git", "express", "mongodb"],
            "min_cgpa": 6.5,
            "degrees": ["B.Tech", "BCA", "MCA", "B.Sc", "Diploma"],
            "min_exp": 1.0,
            "desc": "Develop responsive web applications with JavaScript, React, and Node.js."
        }
    }

ROLE_SPECS = get_role_specs()

def compute_candidate_match(row, target_role, tfidf_model=None):
    specs = ROLE_SPECS.get(target_role)
    if not specs:
        # fallback to first role if not found
        specs = list(ROLE_SPECS.values())[0]

    # candidate data
    cand_skills_str = str(row.get('Skills', '')).lower()
    cand_skills = [s.strip() for s in cand_skills_str.split(',') if s.strip()]
    cand_text = str(row.get('Resume_Text', '')).lower()
    cand_degree = str(row.get('Education', '')).strip()
    cand_cgpa = float(row.get('CGPA', 0.0)) if pd.notnull(row.get('CGPA')) else 7.0
    cand_exp = float(row.get('Experience_Years', 0.0)) if pd.notnull(row.get('Experience_Years')) else 0.0

    # 1. skill match (count overlap against role required skills)
    req_skills = specs['skills']
    matched_skills = []
    missing_skills = []
    for req in req_skills:
        # check both skills column and resume text
        if any(req in cs for cs in cand_skills) or (req in cand_text):
            matched_skills.append(req)
        else:
            missing_skills.append(req)

    skill_match = len(matched_skills) / max(1, len(req_skills))

    # 2. text similarity
    text_sim = 0.5
    if tfidf_model is not None:
        try:
            vecs = tfidf_model.transform([cand_text, specs['desc']])
            sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            text_sim = float(sim)
        except Exception:
            text_sim = skill_match
    else:
        text_sim = skill_match

    # 3. degree and criteria checks
    degree_ok = cand_degree in specs['degrees']
    cgpa_ok = cand_cgpa >= specs['min_cgpa']
    exp_gap = cand_exp - specs['min_exp']

    # model base probability
    model_prob = float(row.get('Shortlist_Probability', 0.5))

    # 4. Hard rules check:
    # If degree not allowed, skill match < 30%, CGPA below min, or experience below min -> hard drop to Not a match
    is_hard_disqualified = False
    disqualify_reason = []
    if not degree_ok:
        is_hard_disqualified = True
        disqualify_reason.append(f"Degree '{cand_degree}' does not meet requirement ({', '.join(specs['degrees'][:3])})")
    if skill_match < 0.30:
        is_hard_disqualified = True
        disqualify_reason.append(f"Low skill match ({int(skill_match*100)}% < 30% min)")
    if not cgpa_ok:
        is_hard_disqualified = True
        disqualify_reason.append(f"CGPA {cand_cgpa:.2f} below minimum requirement ({specs['min_cgpa']:.1f})")
    if exp_gap < 0:
        is_hard_disqualified = True
        disqualify_reason.append(f"Experience {cand_exp:.1f} yrs below minimum ({specs['min_exp']:.1f} yrs)")

    # 5. Composite Final Score: 0.6 * model_prob + 0.3 * skill_match + 0.1 * text_similarity
    composite_score = (0.6 * model_prob) + (0.3 * skill_match) + (0.1 * text_sim)
    
    if is_hard_disqualified:
        # Cap score so disqualified candidates never rank in top tier
        composite_score = min(composite_score * 0.4, 0.35)
        result_label = "Not a match"
    else:
        if composite_score >= 0.70:
            result_label = "Strong match"
        elif composite_score >= 0.45:
            result_label = "Possible match"
        else:
            result_label = "Not a match"

    # 6. Detailed, individualized 'Why' assessment
    matched_disp = ", ".join(matched_skills[:3]) if matched_skills else "None"
    missing_disp = ", ".join(missing_skills[:2]) if missing_skills else "None"
    
    if is_hard_disqualified:
        why = f"Not qualified: {'; '.join(disqualify_reason)}."
    elif result_label == "Strong match":
        why = f"Strong fit: Matched {len(matched_skills)}/{len(req_skills)} skills ({matched_disp}). {cand_exp} yrs exp, CGPA {cand_cgpa:.1f}."
    elif result_label == "Possible match":
        why = f"Moderate fit: Matched {matched_disp}; missing {missing_disp}. CGPA {cand_cgpa:.1f}."
    else:
        why = f"Low relevance: Missing key competencies ({missing_disp})."

    return {
        'final_score': round(composite_score, 4),
        'skill_match': round(skill_match, 4),
        'text_similarity': round(text_sim, 4),
        'result_label': result_label,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'degree_ok': degree_ok,
        'cgpa_ok': cgpa_ok,
        'exp_gap': round(exp_gap, 1),
        'why': why
    }
