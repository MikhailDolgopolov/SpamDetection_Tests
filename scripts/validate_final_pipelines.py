import argparse
import glob
import json
from pathlib import Path

import joblib
from tqdm import tqdm

from src.data_loader import download_dataset, VALIDATION_SET_NAME
from src.utils import measure_inference_metrics


def validate_pipes():
    X_val, y_val = download_dataset(VALIDATION_SET_NAME, 'tuple')

    exps = glob.glob("experiments/*/final")
    for final_dir in tqdm(exps, desc="Validating pipelines"):
        final_dir = Path(final_dir)
        pipe_path = final_dir / "best_pipeline.joblib"
        pipe = joblib.load(pipe_path)

        metrics = measure_inference_metrics(pipe, X_val, y_val)
        with open(final_dir / "inference_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    validate_pipes()
