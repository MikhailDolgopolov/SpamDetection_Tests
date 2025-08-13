import json
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, List, Mapping, Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from src.config import ExperimentConfig
from src.vectorizers.SbertVectorizer import SBERTVectorizer

_vectorizer_registry: Dict[str, type] = {
    "CountVectorizer": CountVectorizer,
    "TfidfVectorizer": TfidfVectorizer,
    "SBERTVectorizer": SBERTVectorizer,
}
_classifier_registry: Dict[str, type] = {
    "MultinomialNB": MultinomialNB,
    "LogisticRegression": LogisticRegression,
    "LinearSVC": LinearSVC,
}


def _is_embedding_vectorizer(name: str) -> bool:
    """Простая эвристика: эти векторизаторы обычно возвращают dense float embeddings."""
    return name.lower() in {"sbert", "fasttext", "doc2vec", "transformers"}


def validate_compatibility(cfg: ExperimentConfig):
    if _is_embedding_vectorizer(cfg.vectorizer) and cfg.classifier == "MultinomialNB":
        raise RuntimeError(
            "Selected vectorizer produces dense float embeddings (SBERT/fasttext/transformers). "
            "MultinomialNB expects non-negative counts/frequencies. "
            "Please choose a classifier like LogisticRegression / SVC / RandomForest, "
            "or switch vectorizer to 'count'/'tfidf'."
        )


def build_param_grid(cfg: ExperimentConfig) -> Dict[str, List[Any]]:
    grid = {}
    for k, v in cfg.vectorizer_params.items():
        grid[f"vec__{k}"] = v
    for k, v in cfg.classifier_params.items():
        grid[f"clf__{k}"] = v
    return grid


def build_pipeline(cfg: ExperimentConfig) -> Pipeline:
    vec_name = cfg.vectorizer
    clf_name = cfg.classifier

    # instantiate base vectorizer and classifier
    vec_cls = _vectorizer_registry[vec_name]
    clf_cls = _classifier_registry[clf_name]

    vec_inst = vec_cls()
    clf_inst = clf_cls()

    return Pipeline([("vec", vec_inst), ("clf", clf_inst)])


def build_grid_search(cfg: ExperimentConfig, verbose=1) -> GridSearchCV:
    pipeline = build_pipeline(cfg)
    param_grid = build_param_grid(cfg)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )


def _build_pipeline_from_params(vec_name: str, clf_name: str, vec_params: Dict[str,Any], clf_params: Dict[str,Any]):
    """
    Recreate pipeline from params dict (maps param->value, no prefixes).
    Example: vec_params = {'min_df': 2, 'ngram_range': (1,2)}
    """
    Vec = _vectorizer_registry[vec_name]
    Clf = _classifier_registry[clf_name]
    vec = Vec()
    clf = Clf()
    for k, v in vec_params.items():
        if 'range' in k and isinstance(v, Iterable):
            vec_params[k] = tuple(v)

    pipe = Pipeline([("vec", vec), ("clf", clf)])
    # set params using pipeline naming
    pipe.set_params(**{f"vec__{k}": v for k, v in vec_params.items()})
    pipe.set_params(**{f"clf__{k}": v for k, v in clf_params.items()})
    return pipe


def build_pipeline_from_experiment_results(experiment_dir: str | Path) -> Optional[Pipeline]:
    filepath = Path(experiment_dir)/"results.json"
    if not filepath.exists():
        print(f"Warning: Experiment result at {filepath} wasn't found")
        return None
    arch = json.load(open(filepath, "r", encoding="utf-8"))["best_arch"]

    vec_name = arch["vec"]["class_name"]
    clf_name = arch["clf"]["class_name"]
    vec_params = arch["vec"]["params"]
    clf_params = arch["clf"]["params"]
    return _build_pipeline_from_params(vec_name, clf_name, vec_params, clf_params)


def dump_pipeline_architecture(pipeline: Pipeline) -> Dict[str, Any]:
    """Return a JSON‑serialisable description of the pipeline."""
    arch = {}
    for name, est in pipeline.named_steps.items():
        # Grab everything that is *not* private and not a function
        params = {
            k: v
            for k, v in vars(est).items()
            if not k.startswith("_") and not k.endswith("_") and not callable(v)
        }
        arch[name] = {
            "class_name": est.__class__.__name__,
            "params": params
        }
    return arch
