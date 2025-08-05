import os
import glob
import re

import pandas as pd
from email import message_from_file
from bs4 import BeautifulSoup # For HTML cleaning

def parse_emails(ham_dir, spam_dir):
    """Parses .eml files from ham and spam directories into a DataFrame.

    Args:
        ham_dir (str): Path to the directory containing ham emails (.eml files).
        spam_dir (str): Path to the directory containing spam emails (.eml files).

    Returns:
        pd.DataFrame: A DataFrame with 'text' and 'label' columns.  'label' is 0 for ham, 1 for spam.
    """

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


def clean_html(html_string):
    """Removes HTML tags from a string."""
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup.get_text()  # Extract text content only


def emails_to_pandas():
    ham_directory = "data/test_task_emails/ham"
    spam_directory = "data/test_task_emails/spam"

    df = parse_emails(ham_directory, spam_directory)
    df.to_csv("data/test_task_emails.csv", index=False)


if __name__ == "__main__":
    emails_to_pandas()