"""
AI Resume Screening System
==========================
Entry point: trains models, generates visualizations, and screens resumes
against sample job descriptions.

Usage:
    python main.py              # full pipeline (train + screen)
    python main.py --screen     # load saved model and screen only
    python main.py --custom     # load saved model + screen inputs/ files only

Text file input:
    Edit  inputs/my_resume.txt       with your resume
    Edit  inputs/job_description.txt with the job you want
    Then run:  python main.py  (or  python main.py --custom  if model already trained)
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

INPUTS_DIR   = os.path.join(BASE_DIR, 'inputs')
RESUME_FILE  = os.path.join(INPUTS_DIR, 'my_resume.txt')
JD_FILE      = os.path.join(INPUTS_DIR, 'job_description.txt')

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.model_trainer import ModelTrainer
from src.resume_screener import ResumeScreener
from src.visualizer import Visualizer

SAMPLE_JOB_DESCRIPTIONS = [
    {
        'title': 'Data Scientist',
        'description': """
            We are looking for a Data Scientist with expertise in Python, machine learning,
            deep learning, TensorFlow, Keras, and NLP. The candidate should have strong skills
            in pandas, numpy, scikit-learn, and data visualization with matplotlib and seaborn.
            Experience with SQL databases and cloud platforms (AWS, GCP) is a plus.
            Must be able to build, evaluate, and deploy predictive models at scale.
        """,
    },
    {
        'title': 'Java Developer',
        'description': """
            Seeking a Java Developer with 3+ years of experience. Must have strong knowledge
            of Spring Boot, Hibernate, Microservices architecture, REST APIs, and Maven.
            Experience with MySQL or PostgreSQL required. Docker and Kubernetes are a plus.
            Must write clean, testable code with JUnit and follow Agile/Scrum practices.
        """,
    },
    {
        'title': 'HR Manager',
        'description': """
            We need an experienced HR Manager with 5+ years in talent acquisition, employee
            relations, performance management, and payroll processing. Proficiency in HRIS
            systems such as SAP HR or Workday is essential. Strong communication skills and
            knowledge of labor laws and compliance regulations required. MBA in HR preferred.
        """,
    },
    {
        'title': 'DevOps Engineer',
        'description': """
            Looking for a DevOps Engineer with hands-on experience in CI/CD pipeline design,
            Docker, Kubernetes, Terraform, and cloud infrastructure (AWS or GCP). Must be
            comfortable with monitoring tools like Prometheus and Grafana. Linux administration
            and scripting (Bash/Python) are required. Experience with GitOps practices is a plus.
        """,
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

PLACEHOLDER_PHRASES = [
    'paste your resume here',
    'paste the job description here',
    'replace this entire file',
]

def _read_input_file(path: str) -> str:
    """Return file content, or empty string if missing / still a template."""
    if not os.path.exists(path):
        return ''
    text = open(path, encoding='utf-8').read().strip()
    lower = text.lower()
    if any(p in lower for p in PLACEHOLDER_PHRASES):
        return ''
    return text


def _load_custom_inputs():
    """Return (resume_text, jd_text) from inputs/ files, or (None, None)."""
    resume = _read_input_file(RESUME_FILE)
    jd     = _read_input_file(JD_FILE)
    return resume or None, jd or None


def _run_custom_screening(screener, visualizer, resume_text, jd_text, df):
    """Screen a custom resume against all resumes AND rank it for a custom JD."""
    print('\n' + '─' * 65)
    print('  CUSTOM FILE INPUT SCREENING')
    print('─' * 65)

    # ── A. Rank your resume against the custom job description ────────
    if jd_text:
        print('\n  [A] How well does YOUR resume match the job description?')
        jd_matches = screener.screen_resumes(jd_text, top_n=10)
        predicted_cat = jd_matches['Predicted_JD_Category'].iloc[0]
        print(f'      Job predicted as category : {predicted_cat}')
        print(f'      Top 10 matching resumes from the database:\n')
        for _, row in jd_matches.iterrows():
            print(f'        #{int(row["Rank"]):>2}  [{row["Category"]:<25}]  '
                  f'{row["Match_%"]:>6.1f}%')
        visualizer.plot_screening_results(jd_matches, 'Your Job Description', save_suffix='_custom_jd')

    # ── B. Where does YOUR resume rank? ───────────────────────────────
    if resume_text and jd_text:
        print('\n  [B] Ranking YOUR resume alongside the database for that job...')
        preprocessor = TextPreprocessor()
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import pandas as pd

        clean_jd     = preprocessor.preprocess(jd_text)
        clean_resume = preprocessor.preprocess(resume_text)

        jd_vec     = screener.tfidf.transform([clean_jd])
        resume_vec = screener.tfidf.transform([clean_resume])

        db_sims   = cosine_similarity(jd_vec, screener.resume_vectors).flatten()
        your_sim  = float(cosine_similarity(jd_vec, resume_vec)[0][0])
        your_pct  = round(your_sim * 100, 2)

        rank = int((db_sims > your_sim).sum()) + 1
        total = len(db_sims) + 1

        print(f'\n      Your resume match score : {your_pct:.1f}%')
        print(f'      Your rank               : #{rank} out of {total} resumes')

        if rank == 1:
            verdict = 'EXCELLENT — you are the top match!'
        elif rank <= total * 0.10:
            verdict = 'GREAT — you are in the top 10%'
        elif rank <= total * 0.25:
            verdict = 'GOOD — you are in the top 25%'
        elif rank <= total * 0.50:
            verdict = 'AVERAGE — you are in the top 50%'
        else:
            verdict = 'BELOW AVERAGE — consider adding more relevant keywords'
        print(f'      Verdict                 : {verdict}')

        # Bar chart: your score vs top 5 from DB
        top5 = jd_matches.head(5).copy()
        your_row = pd.DataFrame([{
            'Rank': 0,
            'Category': 'YOUR RESUME',
            'Match_%': your_pct,
            'Similarity_Score': your_sim,
            'Predicted_JD_Category': predicted_cat if jd_text else '',
            'Resume_Preview': resume_text[:200] + '...',
            'Resume': resume_text,
        }])
        combined = pd.concat([your_row, top5], ignore_index=True)
        combined['Rank'] = combined.index + 1
        visualizer.plot_screening_results(combined, 'Your Resume vs Top Matches', save_suffix='_your_rank')

    # ── C. What category does YOUR resume fall into? ──────────────────
    if resume_text:
        print('\n  [C] What job category does your resume belong to?')
        resume_matches = screener.screen_resumes(resume_text, top_n=5)
        predicted = resume_matches['Predicted_JD_Category'].iloc[0]
        print(f'      Predicted category      : {predicted}')
        print(f'      Top 5 similar resumes in the database:\n')
        for _, row in resume_matches.iterrows():
            print(f'        #{int(row["Rank"]):>2}  [{row["Category"]:<25}]  '
                  f'{row["Match_%"]:>6.1f}%  —  {row["Resume_Preview"][:60]}...')

    print('\n' + '─' * 65)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline():
    print('\n' + '=' * 65)
    print('         AI RESUME SCREENING SYSTEM — Full Pipeline')
    print('=' * 65)

    loader       = DataLoader()
    preprocessor = TextPreprocessor()
    trainer      = ModelTrainer()
    visualizer   = Visualizer()

    # ── 1. Load data ──────────────────────────────────────────────────
    print('\n[STEP 1/5]  Loading dataset')
    df    = loader.load_data()
    stats = loader.get_stats(df)
    print(f"  Total resumes : {stats['total_resumes']}")
    print(f"  Categories    : {stats['num_categories']}")
    print(f"  Avg length    : {stats['avg_resume_length']} chars")

    # ── 2. EDA visualizations ─────────────────────────────────────────
    print('\n[STEP 2/5]  Generating EDA visualizations')
    visualizer.plot_category_distribution(df)
    visualizer.plot_resume_length_distribution(df)
    visualizer.plot_word_frequency(df)
    visualizer.plot_wordcloud(df)

    # ── 3. Preprocess ─────────────────────────────────────────────────
    print('\n[STEP 3/5]  Preprocessing text')
    df = preprocessor.preprocess_dataframe(df)

    # ── 4. Train & evaluate models ────────────────────────────────────
    print('\n[STEP 4/5]  Training ML models')
    results = trainer.train_and_evaluate(df)
    visualizer.plot_model_comparison(results)
    visualizer.plot_confusion_matrix(
        trainer.best_model, trainer.X_test, trainer.y_test, trainer.label_encoder
    )
    visualizer.plot_f1_by_category(results, trainer.best_model_name)
    trainer.save_model()

    # ── 5. Screen sample JDs ──────────────────────────────────────────
    print('\n[STEP 5/5]  Screening resumes for sample job descriptions')
    screener = ResumeScreener(
        trainer.best_model,
        trainer.tfidf_vectorizer,
        df,
        trainer.label_encoder,
        all_resume_vectors=trainer.all_resume_vectors,
    )

    for jd in SAMPLE_JOB_DESCRIPTIONS:
        print(f'\n  Job: {jd["title"]}')
        matches = screener.screen_resumes(jd['description'], top_n=5)
        visualizer.plot_screening_results(matches, jd['title'])
        print(f'  Predicted category: {matches["Predicted_JD_Category"].iloc[0]}')
        print('  Top matches:')
        for _, row in matches.iterrows():
            print(f'    #{int(row["Rank"])}  [{row["Category"]}]  '
                  f'{row["Match_%"]:.1f}%  —  {row["Resume_Preview"][:80]}...')

    # ── 6. Custom file input (auto-detected) ──────────────────────────
    resume_text, jd_text = _load_custom_inputs()
    if resume_text or jd_text:
        _run_custom_screening(screener, visualizer, resume_text, jd_text, df)
    else:
        print('\n  Tip: edit inputs/my_resume.txt and inputs/job_description.txt')
        print('       then re-run to screen your own resume.')

    _print_summary()


def run_screen_only():
    print('\n' + '=' * 65)
    print('         AI RESUME SCREENING SYSTEM — Screen Mode')
    print('=' * 65)

    try:
        model, vectorizer, encoder, vectors = ModelTrainer.load_model()
    except FileNotFoundError:
        print('\n  No saved model found. Run without --screen to train first.')
        sys.exit(1)

    loader       = DataLoader()
    preprocessor = TextPreprocessor()
    visualizer   = Visualizer()

    df = loader.load_data()
    df = preprocessor.preprocess_dataframe(df)

    screener = ResumeScreener(model, vectorizer, df, encoder, all_resume_vectors=vectors)

    for jd in SAMPLE_JOB_DESCRIPTIONS:
        print(f'\n  Job: {jd["title"]}')
        matches = screener.screen_resumes(jd['description'], top_n=5)
        visualizer.plot_screening_results(matches, jd['title'])
        print(f'  Predicted category: {matches["Predicted_JD_Category"].iloc[0]}')
        for _, row in matches.iterrows():
            print(f'    #{int(row["Rank"])}  [{row["Category"]}]  '
                  f'{row["Match_%"]:.1f}%')

    resume_text, jd_text = _load_custom_inputs()
    if resume_text or jd_text:
        _run_custom_screening(screener, visualizer, resume_text, jd_text, df)

    _print_summary()


def run_custom_only():
    print('\n' + '=' * 65)
    print('         AI RESUME SCREENING SYSTEM — Custom Input Mode')
    print('=' * 65)

    resume_text, jd_text = _load_custom_inputs()
    if not resume_text and not jd_text:
        print('\n  inputs/my_resume.txt and inputs/job_description.txt are empty.')
        print('  Replace the placeholder text in those files and try again.')
        sys.exit(1)

    try:
        model, vectorizer, encoder, vectors = ModelTrainer.load_model()
    except FileNotFoundError:
        print('\n  No saved model found. Run  python main.py  first to train.')
        sys.exit(1)

    loader       = DataLoader()
    preprocessor = TextPreprocessor()
    visualizer   = Visualizer()

    df = loader.load_data()
    df = preprocessor.preprocess_dataframe(df)

    screener = ResumeScreener(model, vectorizer, df, encoder, all_resume_vectors=vectors)
    _run_custom_screening(screener, visualizer, resume_text, jd_text, df)
    _print_summary()


def _print_summary():
    outputs = os.path.join(BASE_DIR, 'outputs')
    models  = os.path.join(BASE_DIR, 'models')
    charts  = sorted(os.listdir(outputs)) if os.path.isdir(outputs) else []
    print('\n' + '=' * 65)
    print(f'  Visualizations saved to  outputs/  ({len(charts)} charts)')
    print(f'  Model artifacts saved to models/')
    print('\n  Charts generated:')
    for c in charts:
        print(f'    - {c}')
    print('=' * 65 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Resume Screening System')
    parser.add_argument('--screen', action='store_true',
                        help='Skip training; load saved model and screen only')
    parser.add_argument('--custom', action='store_true',
                        help='Screen only inputs/my_resume.txt against inputs/job_description.txt')
    args = parser.parse_args()

    if args.custom:
        run_custom_only()
    elif args.screen:
        run_screen_only()
    else:
        run_full_pipeline()
