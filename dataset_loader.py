import sys
from typing import Dict, Tuple

import kagglehub
import numpy as np
import pandas as pd

dataset_map = {
    "enron": {"name": "mohinurabdurahimova/maildataset", "file": "mail_data.csv"},
    "lingspam": {"name": "mandygu/lingspam-dataset", "file": "messages.csv"}
}


def enron_tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("")
    df["text"] = df["Message"]
    df["label"] = np.where(df["Category"] == "spam", 1, 0)
    return df[["text", "label"]]


def lingspam_tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("")
    df["text"] = df["subject"] + " " + df["message"]
    return df[["text", "label"]]


preprocesses = {
    "mohinurabdurahimova/maildataset": enron_tidy,
    "mandygu/lingspam-dataset": lingspam_tidy,
}


def download_dataset(name: str) -> Tuple[pd.Series, pd.Series]:
    """
    Downloads a dataset from Kaggle and returns processed text and label columns.

    Args:
        name: Kaggle dataset handle

    Returns:
        Tuple of (text_series, label_series) as pandas Series objects

    Raises:
        KeyError: If column names aren't standardized in preprocesses
        Exception: For other errors (prints error message)
    """
    df_path = dataset_map[name]
    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download(
            df_path['name'],
            path=df_path['file']
        )

        # Read CSV
        df = pd.read_csv(dataset_path)

        # Preprocess
        standard = preprocesses[df_path['name']](df)

        # Return processed columns
        return standard["text"].str.lower(), standard["label"]

    except KeyError as ke:
        if 'name' in str(ke):
            raise KeyError(f"Dataset '{df_path.get('name', 'unknown')}' not found in preprocesses mapping")
        else:
            raise KeyError(f"Column names for {df_path['name']} were not standardized: {str(ke)}")

    except Exception as e:
        print(f"Error processing dataset {df_path.get('name', 'unknown')}: {str(e)}",
              file=sys.stderr)
        return pd.Series(dtype='object'), pd.Series(dtype='object')

