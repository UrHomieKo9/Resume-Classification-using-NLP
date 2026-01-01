import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.models.classifier import ResumeClassifier


def main():
    # Load data
    df = pd.read_csv("data/processed/cleaned_resumes.csv")
    df = df.dropna(subset=["cleaned_text", "Category"])
    df = df[df["cleaned_text"].str.strip() != ""]

    X = df["cleaned_text"].tolist()
    y = df["Category"].tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train classifier (CSV-free training)
    classifier = ResumeClassifier()

    # Manually fit using training data
    classifier.vectorizer.fit(X_train)
    X_train_vec = classifier.vectorizer.transform(X_train)
    classifier.label_encoder.fit(y_train)
    y_train_enc = classifier.label_encoder.transform(y_train)

    classifier.model.fit(X_train_vec, y_train_enc)

    # Predict
    preds = classifier.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
