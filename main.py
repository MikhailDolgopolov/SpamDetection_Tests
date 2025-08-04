import os
import pickle
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dataset_loader import download_dataset

model_dir = Path("models")


def save_model(model, filename="best_model.pkl"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir / Path(filename), "wb") as f:
        pickle.dump(model, f)


def load_model(filename="best_model.pkl"):
    with open(model_dir / Path(filename), "rb") as f:
        return pickle.load(f)


def train_tuned_model(_x_train, _y_train):
    pipeline = Pipeline([('vec', CountVectorizer(stop_words='english')), ('clf', MultinomialNB())])
    param_grid = {
        'vec__min_df': [1],
        'vec__max_df': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'vec__ngram_range': [(1, 2)],
        'clf__alpha': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=4, scoring='f1', n_jobs=4)
    grid.fit(_x_train, _y_train)
    return grid


def evaluate_model(model, _x_val, _y_val):
    y_pred = model.predict(_x_val)
    print(classification_report(_y_val, y_pred, digits=4))


def train_default_model(_x_train, _y_train):
    """Обучает базовую модель с параметрами по умолчанию."""
    pipeline = Pipeline([('vec', CountVectorizer(stop_words='english')), ('clf', MultinomialNB())])
    pipeline.fit(_x_train, _y_train)
    return pipeline


if __name__ == "__main__":
    dataset_name = "enron"  # Выберите набор данных

    texts, labels = download_dataset(dataset_name)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    default_model = train_default_model(X_train, y_train)
    print("=== DEFAULT PARAMETERS ===")
    evaluate_model(default_model, X_val, y_val)
    save_model(default_model, 'default_model.pkl')

    tuned_model = train_tuned_model(X_train, y_train)
    print("Best params:", tuned_model.best_params_)
    evaluate_model(tuned_model, X_val, y_val)
    save_model(tuned_model)

