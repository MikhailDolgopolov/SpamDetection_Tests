import glob
import json
import os
import time
from pathlib import Path
from pprint import pprint

import joblib
from tqdm import tqdm

from src.data_loader import download_and_merge, download_dataset, VALIDATION_SET_NAME
from src.model_builder import build_pipeline_from_experiment_results
from src.utils import measure_inference_metrics


def train_pipes():
    df = download_and_merge(["enron", "lingspam"], 1)
    X, y = df["text"], df["label"]
    exps = glob.glob("experiments/*")
    for exp_dir in tqdm(exps, desc="Fitting pipelines"):
        exp_dir = Path(exp_dir)
        pipe = build_pipeline_from_experiment_results(exp_dir)
        if not pipe:
            continue
        final_dir = exp_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        # --- timing start ---
        t0 = time.perf_counter()
        pipe.fit(X, y)
        t1 = time.perf_counter()

        # Save pipeline
        model_path = final_dir / "best_pipeline.joblib"
        joblib.dump(pipe, model_path)

        # --- metrics ---
        total_time = t1 - t0
        time_per_sample = total_time / len(X)
        file_size = os.path.getsize(model_path) / 1024 / 1024  # in MB

        metrics = {
            "total_training_time_sec": total_time,
            "time_per_sample_sec": time_per_sample,
            "model_file_size_mb": file_size,
            "num_samples": len(X)
        }

        with open(final_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        tqdm.write(f"[{exp_dir.name}] Training done in {total_time:.2f}s "
                   f"({time_per_sample * 1000:.3f} ms/sample), "
                   f"size={file_size:.2f} MB")


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
    validate_pipes()
