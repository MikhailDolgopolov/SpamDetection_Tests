import pandas as pd

from dataset_loader import download_dataset, merged_dataset
from main import load_model, evaluate_model

if __name__ == "__main__":

    test_df = pd.read_csv("data/test_task_emails.csv").dropna()
    # test_df = download_dataset('enron', 'dataframe')
    # test_df = merged_dataset(["enron"])
    texts, labels = test_df["text"].str.lower(), test_df["label"]

    model = load_model("best_model.pkl")
    evaluate_model(model, texts, labels)


