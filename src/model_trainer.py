import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class ModelTrainer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.all_resume_vectors = None
        self.results = {}

    def train_and_evaluate(self, df: pd.DataFrame) -> dict:
        text_col = 'Cleaned_Resume' if 'Cleaned_Resume' in df.columns else 'Resume'
        X = df[text_col].values
        y = self.label_encoder.fit_transform(df['Category'].values)

        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("  Fitting TF-IDF vectorizer...")
        self.X_train = self.tfidf_vectorizer.fit_transform(self.X_train_raw)
        self.X_test = self.tfidf_vectorizer.transform(self.X_test_raw)
        # Keep all resume vectors for cosine-similarity screening
        self.all_resume_vectors = self.tfidf_vectorizer.transform(X)

        min_class_count = int(np.bincount(self.y_train).min())
        cv_folds = max(2, min(5, min_class_count))

        candidates = {
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=min(5, len(df) // 10 or 1), metric='cosine'),
            'Linear SVM': CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0), cv=cv_folds),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'Naive Bayes': MultinomialNB(alpha=0.1),
        }

        best_acc = -1
        for name, model in candidates.items():
            print(f"  Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)

            try:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy')
            except ValueError:
                cv_scores = np.array([acc])
            self.results[name] = {
                'accuracy': acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'report': classification_report(
                    self.y_test, y_pred,
                    target_names=self.label_encoder.classes_,
                    zero_division=0
                ),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'model': model,
            }
            print(f"    Accuracy: {acc:.4f}  |  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            if acc > best_acc:
                best_acc = acc
                self.best_model = model
                self.best_model_name = name

        print(f"\n  Best model: {self.best_model_name} (accuracy={best_acc:.4f})")
        print("\n  Classification Report (best model):")
        print(self.results[self.best_model_name]['report'])
        return self.results

    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
        joblib.dump(self.tfidf_vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        joblib.dump(self.all_resume_vectors, os.path.join(MODELS_DIR, 'all_resume_vectors.pkl'))
        print(f"  Model artifacts saved to {MODELS_DIR}/")

    @staticmethod
    def load_model():
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
        encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        vectors = joblib.load(os.path.join(MODELS_DIR, 'all_resume_vectors.pkl'))
        return model, vectorizer, encoder, vectors
