"""
Resume Scorer — Quick Test
==========================
Run:  python3 test.py

Reads your resume and job description from the inputs/ folder,
scores your resume against the job, and shows your rank.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

RESUME_FILE = os.path.join(BASE_DIR, 'inputs', 'my_resume.txt')
JD_FILE     = os.path.join(BASE_DIR, 'inputs', 'job_description.txt')

from sklearn.metrics.pairwise import cosine_similarity
from src.model_trainer import ModelTrainer
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor


def read_file(path, label):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        sys.exit(1)
    text = open(path, encoding='utf-8').read().strip()
    placeholders = ['paste your resume here', 'paste the job description here', 'replace this entire file']
    if any(p in text.lower() for p in placeholders):
        print(f"\n  {label} is still a template.")
        print(f"  Open  {path}  and replace the placeholder text with your actual content.")
        sys.exit(1)
    return text


def score_resume(resume_text, jd_text):
    print('\n' + '=' * 60)
    print('         RESUME SCORER')
    print('=' * 60)

    # Load model
    print('\n  Loading trained model...')
    try:
        model, vectorizer, encoder, db_vectors = ModelTrainer.load_model()
    except FileNotFoundError:
        print('\n  No trained model found.')
        print('  Run  python3 main.py  first to train the model.')
        sys.exit(1)

    # Load resume database
    print('  Loading resume database...')
    loader = DataLoader()
    preprocessor = TextPreprocessor()
    df = loader.load_data()
    df = preprocessor.preprocess_dataframe(df)

    # Preprocess inputs
    clean_resume = preprocessor.preprocess(resume_text)
    clean_jd     = preprocessor.preprocess(jd_text)

    resume_vec = vectorizer.transform([clean_resume])
    jd_vec     = vectorizer.transform([clean_jd])

    # Predict categories
    resume_cat = encoder.inverse_transform(model.predict(resume_vec))[0]
    jd_cat     = encoder.inverse_transform(model.predict(jd_vec))[0]

    # Score your resume against the job description
    your_score  = float(cosine_similarity(jd_vec, resume_vec)[0][0])
    your_pct    = round(your_score * 100, 2)

    # Score all resumes in DB against the same job
    db_scores = cosine_similarity(jd_vec, db_vectors).flatten()
    rank      = int((db_scores > your_score).sum()) + 1
    total     = len(db_scores) + 1

    # Top DB matches
    import numpy as np
    top_idx    = np.argsort(db_scores)[::-1][:5]
    top_cats   = df['Category'].iloc[top_idx].tolist()
    top_scores = (db_scores[top_idx] * 100).round(2).tolist()

    # Verdict
    pct_rank = rank / total
    if rank == 1:
        verdict = 'EXCELLENT  — you are the #1 match!'
    elif pct_rank <= 0.10:
        verdict = 'GREAT      — top 10%'
    elif pct_rank <= 0.25:
        verdict = 'GOOD       — top 25%'
    elif pct_rank <= 0.50:
        verdict = 'AVERAGE    — top 50%'
    else:
        verdict = 'BELOW AVG  — add more relevant keywords'

    # Print results
    print('\n' + '─' * 60)
    print(f'  Your resume category   : {resume_cat}')
    print(f'  Job description type   : {jd_cat}')
    print('─' * 60)
    print(f'  Match score            : {your_pct:.1f}%')
    print(f'  Your rank              : #{rank} out of {total} resumes')
    print(f'  Verdict                : {verdict}')
    print('─' * 60)
    print('\n  Top 5 matching resumes from the database:')
    for i, (cat, sc) in enumerate(zip(top_cats, top_scores), 1):
        marker = '  <-- you would rank here' if rank == i else ''
        print(f'    #{i}  [{cat:<25}]  {sc:>6.1f}%{marker}')

    if rank > 5:
        print(f'    ...')
        print(f'    #{rank}  [YOUR RESUME             ]  {your_pct:>6.1f}%  <-- your position')

    print()

    # Keyword tips
    jd_words     = set(clean_jd.split())
    resume_words = set(clean_resume.split())
    missing      = jd_words - resume_words
    top_missing  = sorted(missing, key=len, reverse=True)[:10]
    if top_missing:
        print('  Keywords in the job description missing from your resume:')
        print('    ' + ',  '.join(top_missing))
        print('\n  Tip: adding these words may improve your match score.')

    print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    print('\n  Reading inputs...')
    resume_text = read_file(RESUME_FILE, 'my_resume.txt')
    jd_text     = read_file(JD_FILE,     'job_description.txt')
    print(f'  Resume loaded   : {len(resume_text.split())} words')
    print(f'  Job desc loaded : {len(jd_text.split())} words')
    score_resume(resume_text, jd_text)
