#!/usr/bin/env python3
import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import sklearn

from src.config import load_experiment_config, config_to_dict
from src.data_loader import download_and_merge, split
from src.model_builder import build_grid_search, build_pipeline
from src.trainer import fit_and_evaluate, save_model


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


def main(cfg_path: Path):
    cfg = load_experiment_config(cfg_path)

    if not cfg.experiment_name:
        cfg.experiment_name = cfg_path.parent.name

    df = download_and_merge(cfg.datasets, cfg.sample_size)
    X_train, X_val, y_train, y_val = split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state
    )

    gs = build_grid_search(cfg)
    metrics = fit_and_evaluate(gs, X_train, y_train, X_val, y_val, random_state=cfg.random_state)
    save_model(gs.best_estimator_, filename=f"{Path(cfg_path).parent.name}_model.joblib")

    cfg_vec_params = {k:v for k, v in metrics["best_arch"]["vec"].items() if k in
                      {k.split("__")[-1]: v for k,v in cfg.vectorizer_params.items()}}
    cfg_clf_params = {k: v for k, v in metrics["best_arch"]["clf"].items() if k in
                      {k.split("__")[-1]: v for k, v in cfg.classifier_params.items()}}
    readable_arch = {cfg.vectorizer: cfg_vec_params, cfg.classifier: cfg_clf_params}
    metrics["best_arch"] = readable_arch

    clean_metrics = _convert_ndarray_to_list(metrics)
    result_path = Path("experiments") / cfg_path.parent.name / "results.json"
    result_path.write_text(json.dumps(clean_metrics, indent=2))
    print(f"Run finished – metrics written to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path,
                        help="path to run_XXX/config.yaml")
    args = parser.parse_args()
    main(args.config)
