import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.features.tfidf_features import TfidfFeatureExtractor


class ResumeClassifier:
    """
    Logistic Regression based resume category classifier.
    """

    def __init__(self):
        self.vectorizer = TfidfFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )

    def train(self, csv_path: str):
        """
        Train the classifier using cleaned resumes CSV.
        """
        df = pd.read_csv(csv_path)

        # Defensive cleaning
        df = df.dropna(subset=["cleaned_text", "Category"])
        df = df[df["cleaned_text"].str.strip() != ""]

        X_text = df["cleaned_text"].tolist()
        y_labels = df["Category"].tolist()

        X = self.vectorizer.fit_transform(X_text)
        y = self.label_encoder.fit_transform(y_labels)

        self.model.fit(X, y)

    def predict(self, texts):
        """
        Predict categories for given resume texts.
        """
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def evaluate(self, csv_path: str):
        """
        Evaluate classifier performance on the dataset.
        """
        df = pd.read_csv(csv_path)

        df = df.dropna(subset=["cleaned_text", "Category"])
        df = df[df["cleaned_text"].str.strip() != ""]

        X_text = df["cleaned_text"].tolist()
        y_true = df["Category"].tolist()

        X = self.vectorizer.transform(X_text)
        y_true_enc = self.label_encoder.transform(y_true)

        y_pred = self.model.predict(X)

        acc = accuracy_score(y_true_enc, y_pred)
        report = classification_report(
            y_true_enc,
            y_pred,
            target_names=self.label_encoder.classes_
        )

        return acc, report
