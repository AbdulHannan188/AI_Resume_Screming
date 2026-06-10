import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for resource in ('punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab'):
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some domain-relevant words that stopwords would remove
        self.stop_words -= {'not', 'no', 'nor'}

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', ' ', text)         # remove URLs
        text = re.sub(r'\S+@\S+', ' ', text)                 # remove emails
        text = re.sub(r'\+?\d[\d\s\-().]{7,}\d', ' ', text) # remove phone numbers
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)          # remove non-ASCII
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\d+', ' ', text)                     # remove digits
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_filter(self, text: str) -> list:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return tokens

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return ' '.join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("  Cleaning and tokenizing resume text...")
        df['Cleaned_Resume'] = df['Resume'].apply(self.preprocess)
        df['Resume_Length'] = df['Resume'].str.len()
        df['Word_Count'] = df['Resume'].str.split().str.len()
        print(f"  Preprocessing complete. Avg word count: {df['Word_Count'].mean():.0f}")
        return df
