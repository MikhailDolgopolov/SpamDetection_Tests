#!/usr/bin/env python3
import argparse
from pathlib import Path


from src.config import load_experiment_config
from src.data_loader import download_and_merge, split
from src.model_builder import build_grid_search, build_pipeline
from src.trainer import fit_and_evaluate, save_model
from src.utils import save_experiment_artifacts


def main(cfg_path: Path):
    cfg = load_experiment_config(cfg_path)

    if not cfg.own_dir:
        cfg.own_dir = cfg_path.parent

    df = download_and_merge(cfg.datasets, cfg.sample_size)
    X_train, X_val, y_train, y_val = split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state
    )

    gs = build_grid_search(cfg)
    gs, metrics = fit_and_evaluate(gs, X_train, y_train, X_val, y_val, random_state=cfg.random_state)

    save_experiment_artifacts(cfg, gs.best_estimator_, gs.best_params_, metrics)
    # save_model(gs.best_estimator_, filename=f"{Path(cfg_path).parent.name}_model.joblib")
    #

    #
    # clean_metrics = _convert_ndarray_to_list(metrics)
    # result_path = Path("experiments") / cfg_path.parent.name / "results.json"
    # result_path.write_text(json.dumps(clean_metrics, indent=2))
    # print(f"Run finished – metrics written to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path,
                        help="path to run_XXX/config.yaml")
    args = parser.parse_args()
    main(args.config)
