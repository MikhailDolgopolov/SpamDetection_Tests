#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from src.config import load_yaml
from src.data_loader import download_and_merge, split
from src.model_builder import build_grid_search
from src.trainer import fit_and_evaluate, save_model


def main(cfg_path: Path):
    cfg = load_yaml(cfg_path)
    # 1. Data
    df = download_and_merge(cfg.datasets, cfg.test_size)
    X_train, X_val, y_train, y_val = split(df, test_size=0.1,
                                           random_state=cfg.random_state)

    # 2. Model search
    grid = build_grid_search(cfg)
    metrics = fit_and_evaluate(grid, X_train, y_train, X_val, y_val)

    # 3. Persist best model & results
    save_model(grid.best_estimator_, filename=f"{Path(cfg_path).parent.name}_model.pkl")
    result_path = Path("experiments") / cfg_path.parent.name / "results.json"
    result_path.write_text(json.dumps(metrics, indent=2))

    print(f"✅ Run finished – metrics written to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path,
                        help="path to run_XXX/config.yaml")
    args = parser.parse_args()
    main(args.config)
