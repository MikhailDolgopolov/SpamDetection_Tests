import glob
import json
from pathlib import Path
from pprint import pprint

import joblib
from tqdm import tqdm

from src.data_loader import download_and_merge
from src.model_builder import build_pipeline_from_experiment_results
from src.utils import measure_and_save_inference_metrics

if __name__ == "__main__":
    df = download_and_merge(["enron", "lingspam"], 1)
    X, y = df["text"], df["label"]
    exps = glob.glob("experiments/TFIDF*")
    for exp_dir in tqdm(exps, desc="Fitting pipelines"):
        exp_dir = Path(exp_dir)
        pipe = build_pipeline_from_experiment_results(exp_dir)
        if not pipe:
            continue
        (exp_dir / "final").mkdir(parents=True, exist_ok=True)
        pipe.fit(X, y)
        joblib.dump(pipe, exp_dir / "final" / "best_pipeline.joblib")
        # measure metrics on the same X (or a hold-out sample)
        measure_and_save_inference_metrics(pipe, X, y=y, exp_dir=exp_dir, sample_size=5000, per_sample_measure=200)

