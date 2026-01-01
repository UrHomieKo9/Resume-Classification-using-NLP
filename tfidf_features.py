from typing import Tuple
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfFeatureExtractor:
    """
    TF-IDF feature extractor for resume and job description matching.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def compute_similarity(self, resume_vec, job_vec) -> float:
        """
        Computes cosine similarity between two TF-IDF vectors.
        """
        return cosine_similarity(resume_vec, job_vec)[0][0]

    def save(self, path: str):
        joblib.dump(self.vectorizer, path)

    def load(self, path: str):
        self.vectorizer = joblib.load(path)
