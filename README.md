# Resume Classification System

This project builds an end-to-end Natural Language Processing (NLP) system to automatically classify resumes into relevant job categories. The pipeline includes text preprocessing, feature extraction using TF-IDF, and supervised learning with Logistic Regression. The trained model achieves reliable multi-class classification performance and is exposed via a FastAPI backend for real-time predictions. The system demonstrates how machine learning can streamline resume screening and reduce manual effort in hiring workflows.

---

## 🚀 Project Overview

This project solves a real-world problem:  
**Automatically categorizing resumes into relevant job domains** such as Engineering, Finance, Sales, Healthcare, etc.

It demonstrates the complete ML lifecycle:
- Data cleaning & preprocessing
- Feature engineering (TF-IDF)
- Model training & evaluation
- Model persistence
- API-based deployment for real-time inference

---

## 🧠 Problem Statement

Manual resume screening is inefficient and inconsistent at scale. This project builds an NLP-based machine learning system to automatically classify resumes into relevant job categories, improving screening efficiency and accuracy.
 
This system helps automate resume classification to support:
- Recruiters
- HR platforms
- Job portals
- Resume screening tools

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML & NLP:** Scikit-learn, TF-IDF
- **Model:** Logistic Regression (Multiclass)
- **API:** FastAPI
- **Model Persistence:** Joblib

---

## 📂 Project Structure

```bash
job-fit-ml/
│
├── data/
│   ├── raw/                # Original resume dataset
│   └── processed/          # Cleaned resumes
│
├── notebooks/              # EDA & preprocessing validation
│
├── src/
│   ├── preprocessing/      # Text cleaning logic
│   ├── features/           # TF-IDF feature extraction
│   ├── models/             # Classifier & similarity model
│   └── utils/              # Utility functions
│
├── train.py                # Model training script
├── evaluate.py             # Model evaluation script
├── app.py                  # FastAPI application
├── requirements.txt
└── README.md
