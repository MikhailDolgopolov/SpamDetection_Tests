from pprint import pprint
from typing import Dict, Any, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.config import ExperimentConfig

_vectorizer_registry: Dict[str, type] = {
    "count": CountVectorizer,
    "tfidf": TfidfVectorizer,
}


def build_vectorizer(name: str, **kwargs) -> Any:
    cls = _vectorizer_registry[name]
    return cls(**kwargs)


def build_param_grid(cfg) -> Dict[str, List[Any]]:
    return {
        "vec__min_df": cfg.min_df,
        "vec__max_df": cfg.max_df,
        "vec__ngram_range": [tuple(r) for r in cfg.ngram_range],
        "clf__alpha": cfg.alpha,
    }


def build_pipeline() -> Pipeline:
    return Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())])


def build_grid_search(cfg: ExperimentConfig) -> GridSearchCV:
    pipeline = build_pipeline()
    param_grid = build_param_grid(cfg)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1,
        verbose=1,
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
