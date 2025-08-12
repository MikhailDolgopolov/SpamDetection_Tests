import sys, glob, os
from email import message_from_file
from typing import Dict, Tuple, List, Literal

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_map = {
    "enron": {"name": "mohinurabdurahimova/maildataset", "file": "mail_data.csv"},
    "lingspam": {"name": "mandygu/lingspam-dataset", "file": "messages.csv"},
}

VALIDATION_SET = "data/processed/test_task_emails.csv"
VALIDATION_SET_NAME = "task_validation"


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


def download_dataset(name: str, return_type: Literal["tuple", "dataframe"] = "tuple", na_fill=True)\
        -> Tuple[pd.Series, pd.Series] | pd.DataFrame:
    """
    Downloads a dataset from Kaggle and returns processed text and label columns.

    Args:
        name: Kaggle dataset handle
        return_type: "tuple" or "dataframe" - whether to return a tuple of texts and labels, or a dataframe
    Returns:
        Tuple of (text_series, label_series) as pandas Series objects

    Raises:
        KeyError: If column names aren't standardized in preprocesses
        Exception: For other errors (prints error message)
    """
    if name == VALIDATION_SET_NAME:
        df = pd.read_csv(VALIDATION_SET)
        if na_fill:
            df = df.fillna("")
        if return_type == "tuple":
            return df["text"], df["label"]
        elif return_type == "dataframe":
            return df[["text", "label"]]

    df_path = dataset_map[name]
    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download(
            df_path['name'],
            path=df_path['file']
        )

        df = pd.read_csv(dataset_path)
        if na_fill:
            df = df.fillna("")
        # Preprocess
        standard = preprocesses[df_path['name']](df)
        standard['text'] = standard["text"].str.lower()
        # Return processed columns
        if return_type == "tuple":
            return standard["text"], standard["label"]
        elif return_type == "dataframe":
            return standard

    except KeyError as ke:
        if 'name' in str(ke):
            raise KeyError(f"Dataset '{df_path.get('name', 'unknown')}' not found in preprocesses mapping")
        else:
            raise KeyError(f"Column names for {df_path['name']} were not standardized: {str(ke)}")

    except Exception as e:
        print(f"Error processing dataset {df_path.get('name', 'unknown')}: {str(e)}",
              file=sys.stderr)
        return pd.Series(dtype='object'), pd.Series(dtype='object')


def download_and_merge(ds_names: List[str], frac: float=1) -> pd.DataFrame:
    dfs = [download_dataset(name, return_type="dataframe") for name in ds_names]
    return pd.concat(dfs, ignore_index=True).sample(frac=frac)


def parse_emails(ham_dir, spam_dir):
    """Parses .eml files from ham and spam directories into a DataFrame.

    Args:
        ham_dir (str): Path to the directory containing ham emails (.eml files).
        spam_dir (str): Path to the directory containing spam emails (.eml files).

    Returns:
        pd.DataFrame: A DataFrame with 'text' and 'label' columns.  'label' is 0 for ham, 1 for spam.
    """
    import re
    from bs4 import BeautifulSoup

    def clean_html(html_string):
        """Removes HTML tags from a string."""
        soup = BeautifulSoup(html_string, 'html.parser')
        return soup.get_text()  # Extract text content only

    data = []
    for dir_path in [ham_dir, spam_dir]:
        label = 0 if dir_path == ham_dir else 1  # 0 for ham, 1 for spam
        for eml_file in glob.glob(os.path.join(dir_path, "*.eml")):
            try:
                with open(eml_file, 'r') as f:
                    msg = message_from_file(f)

                # Extract the body of the email (text content).  Handle different MIME types.
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get('Content-Disposition'))

                        # Look for plain text parts (most common)
                        if 'text/plain' == ctype:
                            try:
                                body += part.get_payload(decode=True).decode('utf-8', errors='ignore') # Try UTF-8 first
                            except UnicodeDecodeError:
                                try:
                                    body += part.get_payload(decode=True).decode('latin-1', errors='ignore')  # Fallback to latin-1
                                except UnicodeDecodeError:
                                    body += part.get_payload(decode=True).decode('cp1252', errors='ignore') # Another common encoding

                        if 'text/html' == ctype and 'attachment; filename="' not in cdispo:  #Avoid HTML if it's an attachment
                            try:
                                html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                body += clean_html(html_content) # Clean the HTML content
                            except UnicodeDecodeError:
                                try:
                                    html_content = part.get_payload(decode=True).decode('latin-1', errors='ignore')
                                    body += clean_html(html_content)
                                except UnicodeDecodeError:
                                    pass # Skip HTML if decoding fails

                else:
                    try:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')  # Try UTF-8 first
                    except UnicodeDecodeError:
                        try:
                            body = msg.get_payload(decode=True).decode('latin-1', errors='ignore') # Fallback to latin-1
                        except UnicodeDecodeError:
                            body = msg.get_payload(decode=True).decode('cp1252', errors='ignore')
                text = re.sub('(?:\n>?)+', ' ', clean_html(body))
                data.append({'text': text, 'label': label})

            except Exception as e:
                print(f"Error parsing {eml_file}: {e}")  # Handle file reading/parsing errors gracefully
                continue # Skip to the next file if there's an error

    return pd.DataFrame(data)


def emails_to_pandas():
    ham_directory = "data/test_task_emails/ham"
    spam_directory = "data/test_task_emails/spam"

    df = parse_emails(ham_directory, spam_directory)
    df.to_csv(VALIDATION_SET, index=False)


def split(df: pd.DataFrame, test_size=0.1, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"],
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )
    return X_train, X_val, y_train, y_val
