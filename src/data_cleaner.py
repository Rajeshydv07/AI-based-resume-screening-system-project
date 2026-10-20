import os
import re
import pandas as pd
import numpy as np

# text preprocessing setup
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
except:
    # fallback if nltk is not installed
    stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'can', 'will', 'just', 'don', 'should', 'now'
    ])

    class SimpleLemmatizer:
        def lemmatize(self, w):
            if w.endswith('ing') and len(w) > 5:
                return w[:-3]
            if w.endswith('ed') and len(w) > 4:
                return w[:-2]
            if w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
                return w[:-1]
            return w

    lemmatizer = SimpleLemmatizer()

# remove duplicates
def remove_duplicates(df):
    old_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = old_len - len(df)
    return df, dropped

# fill missing values
def fill_missing(df):
    # median for numerical
    num_cols = ['Experience_Years', 'CGPA', 'Projects_Count']
    for c in num_cols:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # mode for categorical
    cat_cols = ['Education', 'Skills', 'Job_Role']
    for c in cat_cols:
        if c in df.columns:
            mode_val = df[c].mode()[0]
            df[c] = df[c].fillna(mode_val)

    return df

# cap outliers using iqr
def cap_outliers(df):
    num_cols = ['Experience_Years', 'CGPA', 'Projects_Count']
    bounds = {}
    for c in num_cols:
        if c in df.columns:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr

            # sensible lower and upper limits
            if c == 'Experience_Years':
                low = max(0.0, low)
            elif c == 'CGPA':
                low = max(0.0, low)
                high = min(10.0, high)
            elif c == 'Projects_Count':
                low = max(0.0, low)

            bounds[c] = (round(low, 2), round(high, 2))
            df[c] = df[c].clip(lower=low, upper=high)

    return df, bounds

# clean text column
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # lowercase
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\S+', '', text)
    # remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # remove stopwords and lemmatize
    words = text.split()
    clean_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(clean_words)

# clean full dataset
def clean_dataset(in_path, out_path):
    df = pd.read_csv(in_path)
    
    # 1. duplicates
    df, dropped = remove_duplicates(df)
    
    # 2. missing values
    df = fill_missing(df)
    
    # 3. outliers
    df, bounds = cap_outliers(df)
    
    # 4. text cleaning
    if 'Resume_Text' in df.columns:
        df['Resume_Text'] = df['Resume_Text'].apply(clean_text)
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df

def main():
    in_path = os.path.join('data', 'raw_resumes.csv')
    out_path = os.path.join('data', 'cleaned_resumes.csv')
    clean_df = clean_dataset(in_path, out_path)
    print("Cleaned data saved to:", out_path)
    print("Final shape:", clean_df.shape)

if __name__ == '__main__':
    main()
