import re
import os
import pandas as pd


# --- PDF text extraction ---

def extract_pdf_text(uploaded_file):
    """
    Extract text from a PDF file object.
    Tries pdfplumber first, falls back to pypdf.
    Returns (text_string, error_message).
    error_message is "" on success, friendly string on failure.
    """
    text = ""

    # try pdfplumber first
    try:
        import pdfplumber
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = " ".join(pages).strip()
        if text:
            return text, ""
    except Exception:
        pass

    # fallback: pypdf
    try:
        import pypdf
        uploaded_file.seek(0)
        reader = pypdf.PdfReader(uploaded_file)
        pages = [p.extract_text() or "" for p in reader.pages]
        text = " ".join(pages).strip()
        if text:
            return text, ""
    except Exception as e:
        return "", f"Could not read PDF ({e})"

    # both succeeded but text is empty = scanned/image PDF
    if not text:
        return "", "This PDF has no readable text. Try a text-based PDF or paste the resume."

    return text, ""


# --- Field extraction from resume text ---

def load_all_skills():
    """Load skill keywords from job_roles.csv. Fallback to a small default list."""
    roles_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'job_roles.csv')
    skills_set = set()
    if os.path.exists(roles_path):
        try:
            df = pd.read_csv(roles_path)
            for val in df['Required_Skills'].dropna():
                for s in str(val).split(','):
                    s = s.strip().lower()
                    if s:
                        skills_set.add(s)
        except Exception:
            pass
    if not skills_set:
        # small fallback
        skills_set = {
            "python", "java", "c++", "sql", "javascript", "react", "node.js",
            "machine learning", "deep learning", "pandas", "numpy", "scikit-learn",
            "docker", "aws", "azure", "kubernetes", "git", "html", "css",
            "rest apis", "system design", "data structures", "algorithms",
            "tensorflow", "pytorch", "linux", "mongodb", "excel", "power bi"
        }
    return skills_set


ALL_SKILLS = load_all_skills()

# known degree keywords
DEGREE_PATTERNS = [
    r'\bPhD\b', r'\bM\.Tech\b', r'\bMTech\b', r'\bM\.Sc\b', r'\bMSc\b',
    r'\bMCA\b', r'\bM\.E\b', r'\bMBA\b',
    r'\bB\.Tech\b', r'\bBTech\b', r'\bB\.E\b', r'\bBCA\b',
    r'\bB\.Sc\b', r'\bBSc\b', r'\bDiploma\b',
]

# canonical display names
DEGREE_CANONICAL = {
    'phd': 'PhD', 'm.tech': 'M.Tech', 'mtech': 'M.Tech',
    'm.sc': 'M.Sc', 'msc': 'M.Sc', 'mca': 'MCA', 'm.e': 'M.Tech',
    'mba': 'MCA',
    'b.tech': 'B.Tech', 'btech': 'B.Tech', 'b.e': 'B.Tech',
    'bca': 'BCA', 'b.sc': 'B.Sc', 'bsc': 'B.Sc', 'diploma': 'Diploma',
}


def extract_resume_fields(text):
    """
    Extract degree, CGPA, experience years, and matched skills from raw resume text.
    Returns a dict with keys: skills, degree, cgpa, experience_years.
    Values are the found value or "Not found" string.
    """
    t = text.lower()

    # 1. Skills: check each known skill keyword against text
    matched_skills = [s for s in ALL_SKILLS if s in t]
    matched_skills.sort()

    # 2. Degree: match from longest (most specific) first
    found_degree = "Not found"
    for pat in DEGREE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(0).lower().rstrip('.')
            found_degree = DEGREE_CANONICAL.get(raw, m.group(0))
            break

    # 3. CGPA: patterns like "CGPA: 8.4", "GPA 8.4", "8.4/10", "8.40 / 10"
    found_cgpa = "Not found"
    cgpa_patterns = [
        r'cgpa[\s:]+(\d+\.?\d*)',
        r'gpa[\s:]+(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*/\s*10',
        r'(\d+\.?\d*)\s*/\s*4',   # 4.0 scale
    ]
    for pat in cgpa_patterns:
        m = re.search(pat, t)
        if m:
            val = float(m.group(1))
            # convert 4-scale to 10-scale roughly
            if '/4' in pat and val <= 4.0:
                val = round(val * 2.5, 2)
            if 0.0 < val <= 10.0:
                found_cgpa = str(round(val, 2))
                break

    # 4. Experience years: "3 years", "3+ years", "3 yrs", "3.5 years of experience"
    found_exp = "Not found"
    exp_patterns = [
        r'(\d+\.?\d*)\+?\s*years?\s+of\s+experience',
        r'(\d+\.?\d*)\+?\s*yrs?\s+of\s+experience',
        r'experience\s+of\s+(\d+\.?\d*)\+?\s*years?',
        r'(\d+\.?\d*)\+?\s*years?\s+experience',
        r'(\d+\.?\d*)\+?\s*years?',
        r'(\d+\.?\d*)\+?\s*yrs?',
    ]
    for pat in exp_patterns:
        m = re.search(pat, t)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 40:
                found_exp = str(val)
                break

    return {
        'skills': ', '.join(matched_skills) if matched_skills else "Not found",
        'degree': found_degree,
        'cgpa': found_cgpa,
        'experience_years': found_exp,
        'matched_skills_list': matched_skills,
    }
