import re
import string
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required resources (run once)
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


class TextCleaner:
    """
    Cleans resume and job description text for NLP tasks.
    """

    def __init__(self, min_length: int = 100, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def _remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def _remove_phone_numbers(self, text: str) -> str:
        return re.sub(r"\+?\d[\d\s\-()]{7,}\d", " ", text)

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", " ", text)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _remove_special_characters(self, text: str) -> str:
        text = text.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"[^a-zA-Z\s]", " ", text)

    def _lemmatize(self, text: str) -> str:
        tokens = nltk.word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(tokens)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self._remove_emails(text)
        text = self._remove_phone_numbers(text)
        text = self._remove_urls(text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = self._remove_special_characters(text)
        text = self._normalize_whitespace(text)

        # Truncate long resumes
        text = text[: self.max_length]

        # Lemmatization
        text = self._lemmatize(text)

        # Drop very short texts
        if len(text) < self.min_length:
            return ""

        return text


def clean_resume_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Applies text cleaning to a dataframe column.
    """
    cleaner = TextCleaner()
    df = df.copy()
    df["cleaned_text"] = df[text_column].apply(cleaner.clean_text)
    df = df[df["cleaned_text"] != ""]
    df.reset_index(drop=True, inplace=True)
    return df
