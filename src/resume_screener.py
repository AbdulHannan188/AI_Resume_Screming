import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from .preprocessor import TextPreprocessor


class ResumeScreener:
    """
    Given a job description, ranks all resumes by cosine similarity
    and optionally filters by predicted job category.
    """

    def __init__(self, model, tfidf_vectorizer: TfidfVectorizer,
                 df: pd.DataFrame, label_encoder: LabelEncoder,
                 all_resume_vectors=None):
        self.model = model
        self.tfidf = tfidf_vectorizer
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.preprocessor = TextPreprocessor()

        text_col = 'Cleaned_Resume' if 'Cleaned_Resume' in df.columns else 'Resume'
        if all_resume_vectors is not None:
            self.resume_vectors = all_resume_vectors
        else:
            self.resume_vectors = self.tfidf.transform(self.df[text_col].values)

    def screen_resumes(self, job_description: str, top_n: int = 10,
                       filter_by_category: bool = False) -> pd.DataFrame:
        clean_jd = self.preprocessor.preprocess(job_description)
        jd_vector = self.tfidf.transform([clean_jd])

        # Predict category of the JD
        predicted_label = self.model.predict(jd_vector)[0]
        predicted_category = self.label_encoder.inverse_transform([predicted_label])[0]

        # Cosine similarity between JD and all resumes
        similarities = cosine_similarity(jd_vector, self.resume_vectors).flatten()

        results = self.df.copy()
        results['Similarity_Score'] = similarities
        results['Predicted_JD_Category'] = predicted_category

        if filter_by_category:
            cat_mask = results['Category'] == predicted_category
            if cat_mask.sum() >= top_n:
                results = results[cat_mask]

        results = results.sort_values('Similarity_Score', ascending=False)
        results = results.head(top_n).reset_index(drop=True)
        results['Rank'] = results.index + 1
        results['Match_%'] = (results['Similarity_Score'] * 100).round(2)

        # Short preview of resume text
        results['Resume_Preview'] = results['Resume'].str[:200] + '...'

        return results[['Rank', 'Category', 'Match_%', 'Similarity_Score',
                         'Predicted_JD_Category', 'Resume_Preview', 'Resume']]

    def get_category_summary(self, screening_results: pd.DataFrame) -> pd.Series:
        return screening_results['Category'].value_counts()
