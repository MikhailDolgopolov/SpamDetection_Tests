from pathlib import Path
from pprint import pprint
from typing import Dict, Any, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.config import ExperimentConfig
from src.vectorizers.SbertVectorizer import SBERTVectorizer

_vectorizer_registry: Dict[str, type] = {
    "count": CountVectorizer,
    "tfidf": TfidfVectorizer,
    "sbert": SBERTVectorizer,
}
_classifier_registry: Dict[str, type] = {
    "MultinomialNB": MultinomialNB,
    "LogisticRegression": LogisticRegression,
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
    grid = dict()
    vec_params = {
        "vec__"+k.split('__')[-1]: v
        for k, v in cfg.vectorizer_params.items()
        if k.startswith(f"{cfg.vectorizer}__")
    }

    grid.update(vec_params)

    clf_params = {
        "clf__"+k.split('__')[-1]: v
        for k, v in cfg.classifier_params.items()
        if k.startswith(f"{cfg.classifier}__")
    }
    grid.update(clf_params)

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


def build_grid_search(cfg: ExperimentConfig) -> GridSearchCV:
    validate_compatibility(cfg)

    pipeline = build_pipeline(cfg)
    param_grid = build_param_grid(cfg)
    # pprint(param_grid)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1,
        verbose=1,
    )


def build_pipeline_from_params(vec_name: str, clf_name: str, vec_params: Dict[str,Any], clf_params: Dict[str,Any]):
    Vec = _vectorizer_registry[vec_name]
    Clf = _classifier_registry[clf_name]
    vec = Vec(**{k:v for k,v in vec_params.items() if k in Vec().__dict__ or True})
    clf = Clf(**{k:v for k,v in clf_params.items() if k in Clf().__dict__ or True})
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    # set remaining params (GridSearch style) in case some params are meant to be set after instantiation
    pipe.set_params(**{f"vec__{k}": v for k,v in vec_params.items()})
    pipe.set_params(**{f"clf__{k}": v for k,v in clf_params.items()})
    return pipe


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
