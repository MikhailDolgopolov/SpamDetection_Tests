import os
import pickle
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dataset_loader import download_dataset, merged_dataset

model_dir = Path("models")


def save_model(model:Pipeline|GridSearchCV, filename="best_model.pkl"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir / Path(filename), "wb") as f:
        pickle.dump(model, f)


def load_model(filename="best_model.pkl"):
    with open(model_dir / Path(filename), "rb") as f:
        return pickle.load(f)


def search_tuned_model(_x_train, _y_train):
    pipeline = Pipeline([('vec', CountVectorizer(stop_words='english')), ('clf', MultinomialNB())])
    param_grid = {
        'vec': [CountVectorizer(stop_words='english'), TfidfVectorizer(stop_words='english')],
        'vec__min_df': [3, 5, 7],
        'vec__max_df': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'vec__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 0.4, 0.7, 1.0]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(_x_train, _y_train)
    return grid


def evaluate_model(model:Pipeline|GridSearchCV, _x_val, _y_val):
    y_pred = model.predict(_x_val)
    print(classification_report(_y_val, y_pred, digits=4))


def train_default_model(_x_train, _y_train):
    """Обучает базовую модель с параметрами по умолчанию."""
    pipeline = Pipeline([('vec', CountVectorizer(stop_words='english')), ('clf', MultinomialNB())])
    pipeline.fit(_x_train, _y_train)
    return pipeline


def general_grid_search(big_df: pd.DataFrame):
    texts, labels = big_df['text'], big_df['label']
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )

    tuned_model = search_tuned_model(X_train, y_train)

    best_params = tuned_model.best_params_
    print("Best params:", best_params)
    # Преобразование словаря для сохранения в JSON
    json_params = {}
    for key, value in best_params.items():
        if '__' in key:
            json_params[key] = value  # Просто копируем значения
        else:
            json_params[key] = type(value).__name__

    # Сохранение параметров в JSON файл
    with open("pipeline_config.json", "w") as f:
        json.dump(json_params, f, indent=2)

    evaluate_model(tuned_model, X_val, y_val)


def load_unfit_best_pipeline():
    loaded_params = json.load(open("pipeline_config.json", "r"))
    vec_params = {}
    clf_params = {}

    for key, value in loaded_params.items():
        if "range" in key:
            value = tuple(value)  # json не поддерживает (кортежи)
        if key.startswith("vec__"):
            vec_params[key.replace("vec__", "")] = value
        elif key.startswith("clf__"):
            clf_params[key.replace("clf__", "")] = value
    pipe = []
    if 'Count' in loaded_params['vec']:
        pipe.append(('vec', CountVectorizer(**vec_params)))
    elif 'Tfidf' in loaded_params['vec']:
        pipe.append(('vec', TfidfVectorizer(**vec_params)))

    pipe.append(('clf', MultinomialNB(**clf_params)))

    return Pipeline(pipe)


def fit_configured_pipeline(big_df: pd.DataFrame) -> Pipeline:
    pipeline = load_unfit_best_pipeline()
    pipeline.fit(big_df["text"], big_df['label'])
    return pipeline


if __name__ == "__main__":
    df = merged_dataset(["enron", "lingspam"], 1)
    # general_grid_search(df)
    save_model(fit_configured_pipeline(df))


