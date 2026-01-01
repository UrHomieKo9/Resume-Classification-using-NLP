import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath("."))

from src.models.similarity_model import ResumeRanker
from src.models.classifier import ResumeClassifier
from src.utils.file_io import load_pickle


def main():
    print("\n=== Job-Fit ML Application ===\n")

    # -------- Resume Ranking Engine --------
    ranker = ResumeRanker("data/processed/cleaned_resumes.csv")

    # -------- Resume Classifier (load trained artifacts) --------
    classifier = ResumeClassifier()
    classifier.model = load_pickle("data/processed/resume_classifier_model.pkl")
    classifier.vectorizer.vectorizer = load_pickle("data/processed/tfidf_vectorizer.pkl")
    classifier.label_encoder = load_pickle("data/processed/label_encoder.pkl")

    while True:
        print("\nChoose an option:")
        print("1. Rank resumes for a job description")
        print("2. Classify a resume")
        print("3. Exit")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            job_desc = input("\nEnter job description:\n")
            results = ranker.rank(job_desc, top_n=5)

            print("\nTop 5 matching resumes:")
            for resume_id, category, score in results:
                print(f"ID: {resume_id} | Category: {category} | Similarity: {score:.3f}")

        elif choice == "2":
            resume_text = input("\nEnter resume text:\n")
            prediction = classifier.predict([resume_text])[0]
            print(f"\nPredicted Resume Category: {prediction}")

        elif choice == "3":
            print("\nExiting application. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()



from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained artifacts
model = joblib.load("data/processed/resume_classifier_model.pkl")
vectorizer = joblib.load("data/processed/tfidf_vectorizer.pkl")
label_encoder = joblib.load("data/processed/label_encoder.pkl")


app = FastAPI(title="Job Fit ML API")

class ResumeInput(BaseModel):
    text: str

@app.post("/predict")
def predict_category(resume: ResumeInput):
    X = vectorizer.transform([resume.text])
    pred = model.predict(X)
    category = label_encoder.inverse_transform(pred)[0]
    return {"predicted_category": category}
