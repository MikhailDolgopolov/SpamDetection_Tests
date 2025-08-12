import json

from sklearn.metrics import classification_report

from src.data_loader import download_dataset, VALIDATION_SET_NAME, split
from src.model_builder import build_pipeline_from_experiment_results

if __name__=="__main__":
    pipe = build_pipeline_from_experiment_results("experiments/TFIDF_NB")

    df = download_dataset(VALIDATION_SET_NAME, return_type="dataframe").fillna("")

    X_train, X_val, y_train, y_val = split(df, test_size=0.1)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_pred))