# src/utils/file_io.py

import pandas as pd
import joblib
import os

def read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame to CSV.
    """
    df.to_csv(path, index=False)
    print(f"Saved CSV to {path}")

def save_pickle(obj, path: str) -> None:
    """
    Save a Python object to a pickle file using joblib.
    """
    joblib.dump(obj, path)
    print(f"Saved object to {path}")

def load_pickle(path: str):
    """
    Load a Python object from a pickle file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    return joblib.load(path)
