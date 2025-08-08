from pprint import pprint
from typing import Dict, Any, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.FastTextVectorizer import FastTextVectorizer
from src.config import ExperimentConfig

_vectorizer_registry: Dict[str, type] = {
    "count": CountVectorizer,
    "tfidf": TfidfVectorizer,
    "fasttext": FastTextVectorizer,
}
_classifier_registry: Dict[str, type] = {
    "MultinomialNB": MultinomialNB,
    "LogisticRegression": LogisticRegression,
}


def build_param_grid(cfg: ExperimentConfig) -> Dict[str, List[Any]]:
    grid = dict()
    vec_params = {"vec__"+k.split('__')[-1]: v for k, v in cfg.vectorizer_params.items() if f"{cfg.vectorizer}__" in k}

    grid.update(vec_params)

    clf_params = {"clf__"+k.split('__')[-1]: v for k, v in cfg.classifier_params.items() if f"{cfg.classifier}__" in k}
    grid.update(clf_params)

    return grid


def build_pipeline(cfg: ExperimentConfig) -> Pipeline:
    vec_name = cfg.vectorizer
    clf_name = cfg.classifier
    return Pipeline([("vec", _vectorizer_registry[vec_name]()), ("clf", _classifier_registry[clf_name]())])


def build_grid_search(cfg: ExperimentConfig) -> GridSearchCV:
    pipeline = build_pipeline(cfg)
    param_grid = build_param_grid(cfg)
    pprint(param_grid)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1,
        verbose=2,
    )


def dump_pipeline_architecture(pipeline: Pipeline) -> Dict[str, Any]:
    """Return a JSON‑serialisable description of the pipeline."""
    arch = {}
    for name, est in pipeline.named_steps.items():
        # Grab everything that is *not* private and not a function
        params = {
            k: v
            for k, v in vars(est).items()
            if not k.startswith("_") and not callable(v)
        }
        arch[name] = {
            "class": est.__class__.__name__,
            **params,
        }
    return arch
