#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.config import load_experiment_config
from src.data_loader import download_and_merge, split
from src.model_builder import build_grid_search
from src.trainer import fit_and_evaluate
from src.utils import save_experiment_artifacts


def main(cfg_path: Path, verbose: int, workers: int, output_file: str):
    """Main function to run the experiment."""
    cfg = load_experiment_config(cfg_path)

    if not cfg.own_dir:
        cfg.own_dir = cfg_path.parent
    if output_file is None:
        output_file = cfg_path.with_name(cfg_path.stem + "_results").with_suffix(".json")
    else:
        output_file = cfg_path.parent / Path(output_file)

    df = download_and_merge(cfg.datasets, cfg.sample_size)
    X_train, X_val, y_train, y_val = split(df, test_size=cfg.test_size, random_state=cfg.random_state)
    gs = build_grid_search(cfg, verbose=verbose, n_jobs=workers)
    gs, fit_results = fit_and_evaluate(gs, X_train, y_train, X_val, y_val, random_state=cfg.random_state)
    save_experiment_artifacts(results=fit_results, output_file=output_file)
    print(f"Run finished. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a spam classification experiment.")
    parser.add_argument("config", type=Path, help="path to an experiment config.yaml")
    parser.add_argument("--verbose", type=int, default=1, help="verbosity level (default: 1)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="CPU cores to use (-1 for all cores; default: -1)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="path to output arch and metrics (default: <config_name>-result.json)")

    args = parser.parse_args()

    main(args.config, verbose=args.verbose, workers=args.n_jobs, output_file=args.output_file)
