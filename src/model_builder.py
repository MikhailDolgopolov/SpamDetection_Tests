from pprint import pprint

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.config import PipelineConfig
from src.feature_engineer import build_vectorizer


def build_grid_search(cfg: PipelineConfig) -> GridSearchCV:
    vec = build_vectorizer(cfg)
    clf = MultinomialNB()

    pipeline = Pipeline([("vec", None), ("clf", clf)])

    # Note that each key points to a *list* of candidate values.
    param_grid = {
        "vec": vec,
        "vec__min_df": cfg.min_df,
        "vec__max_df": cfg.max_df,
        "vec__ngram_range": [tuple(r) for r in cfg.ngram_range],
        "clf__alpha": cfg.alpha
    }


    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1,
        verbose=1  # set to 2 for debugging
    )
