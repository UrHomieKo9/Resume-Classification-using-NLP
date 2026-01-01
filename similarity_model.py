import pandas as pd
from src.features.tfidf_features import TfidfFeatureExtractor

class ResumeRanker:
    """
    Ranks resumes based on similarity to a given job description.
    """

    def __init__(self, resumes_csv_path: str):
        """
        Initialize the ranker with resumes CSV path.
        """
        self.df = pd.read_csv(resumes_csv_path)

        # Defensive cleaning: remove NaN or empty cleaned_text
        self.df = self.df.dropna(subset=["cleaned_text"])
        self.df = self.df[self.df["cleaned_text"].str.strip() != ""]

        self.texts = self.df["cleaned_text"].tolist()
        self.vectorizer = TfidfFeatureExtractor()
        self.resume_vectors = self.vectorizer.fit_transform(self.texts)


    def rank(self, job_description: str, top_n: int = 5):
        """
        Returns top_n resumes most similar to the job description.
        """
        job_vec = self.vectorizer.transform([job_description])
        scores = []

        for idx, resume_vec in enumerate(self.resume_vectors):
            score = self.vectorizer.compute_similarity(resume_vec, job_vec)
            scores.append((self.df.iloc[idx]["ID"], self.df.iloc[idx]["Category"], score))

        # Sort descending by score
        scores.sort(key=lambda x: x[2], reverse=True)

        return scores[:top_n]
