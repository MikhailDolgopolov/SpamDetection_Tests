#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.config import load_experiment_config
from src.data_loader import download_and_merge, split
from src.model_builder import build_grid_search
from src.trainer import fit_and_evaluate
from src.utils import save_experiment_artifacts


def main(cfg_path: Path, verbose: int=1):
    cfg = load_experiment_config(cfg_path)

    # own_dir: folder containing config
    if not cfg.own_dir:
        cfg.own_dir = cfg_path.parent

    df = download_and_merge(cfg.datasets, cfg.sample_size)
    X_train, X_val, y_train, y_val = split(df, test_size=cfg.test_size, random_state=cfg.random_state)

    gs = build_grid_search(cfg, verbose=verbose)
    gs, fit_results = fit_and_evaluate(gs, X_train, y_train, X_val, y_val, random_state=cfg.random_state)

    save_experiment_artifacts(cfg, gs.best_estimator_, fit_results)
    print(f"Run finished. Artifacts saved to experiments/{cfg.own_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="path to run_XXX/config.yaml")
    parser.add_argument("--verbose", type=int, default=1, help="verbosity level")
    args = parser.parse_args()
    main(args.config, verbose=args.verbose)
