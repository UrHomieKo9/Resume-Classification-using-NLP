import pandas as pd
import joblib
from src.models.classifier import ResumeClassifier

def main():
    # Path to processed CSV
    csv_path = "data/processed/cleaned_resumes.csv"

    # Initialize classifier
    classifier = ResumeClassifier()

    # Train classifier
    classifier.train(csv_path)
    print("Training completed successfully!")

    # Save the trained model
    joblib.dump(classifier.model, "data/processed/resume_classifier_model.pkl")
    joblib.dump(classifier.vectorizer, "data/processed/tfidf_vectorizer.pkl")
    joblib.dump(classifier.label_encoder, "data/processed/label_encoder.pkl")
    print("Trained model, vectorizer, and label encoder saved successfully!")

if __name__ == "__main__":
    main()
