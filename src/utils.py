import json
import joblib
import sklearn
import yaml
from pathlib import Path
from typing import Dict, Any, Mapping

from src.config import ExperimentConfig


def _convert_ndarray_to_list(o):
    """Recurse through a dict/list and replace np.ndarray with list."""
    if isinstance(o, Mapping):                     # dict-like
        return {k: _convert_ndarray_to_list(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        return [_convert_ndarray_to_list(i) for i in o]
    elif isinstance(o, sklearn.base.BaseEstimator):
        return type(o).__name__
    else:
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()                 # ← вот где конвертируем
        except ImportError:
            pass
        return o


def save_experiment_artifacts(experiment: ExperimentConfig,
                              best_estimator,
                              best_params: Dict[str, Any],
                              results: Dict[str, Any],
                              save_yaml_results: bool = True):
    experiment.own_dir.mkdir(parents=True, exist_ok=True)

    cfg_vec_params = {k:v for k, v in results["best_arch"]["vec"].items() if k in
                      {k.split("__")[-1]: v for k,v in experiment.vectorizer_params.items()}}
    cfg_clf_params = {k: v for k, v in results["best_arch"]["clf"].items() if k in
                      {k.split("__")[-1]: v for k, v in experiment.classifier_params.items()}}
    readable_arch = {experiment.vectorizer: cfg_vec_params, experiment.classifier: cfg_clf_params}
    results["best_arch"] = readable_arch

    best_params = _convert_ndarray_to_list(best_params)
    results = _convert_ndarray_to_list(results)
    # model
    joblib.dump(best_estimator, experiment.own_dir / "best_pipeline.joblib")
    # best params
    with open(experiment.own_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    # results
    with open(experiment.own_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if save_yaml_results:
        with open(experiment.own_dir / "results.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, sort_keys=False, allow_unicode=True)

    print(f"Run finished – artifacts saved to {experiment.own_dir}")


def split_params(best_params: Dict[str, Any]):
    vec_params = {}
    clf_params = {}
    for k, v in best_params.items():
        if k.startswith("vec__"):
            vec_params[k.replace("vec__", "")] = v
        elif k.startswith("clf__"):
            clf_params[k.replace("clf__", "")] = v

    return vec_params, clf_params

