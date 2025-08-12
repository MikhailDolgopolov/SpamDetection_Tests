import pickle
import tempfile
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.model_builder import dump_pipeline_architecture
from src.utils import _measure_fn, _model_filesize_bytes

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _measure_per_sample_latency(pipeline, X_val, nsamples=100):
    """Измерить latency по выборке отдельных predict([x]) — вернуть median и p95."""
    n = min(nsamples, len(X_val))
    if n == 0:
        return {"median": None, "p95": None, "samples_measured": 0}
    idx = np.random.choice(len(X_val), size=n, replace=False)
    latencies = []
    for i in idx:
        x = [X_val[i]]
        t0 = time.perf_counter()
        try:
            pipeline.predict(x)
        except Exception:
            # если predict падает на единичном элементе, пропускаем
            continue
        latencies.append(time.perf_counter() - t0)
    if not latencies:
        return {"median": None, "p95": None, "samples_measured": 0}
    a = np.array(latencies)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "samples_measured": len(a)}


def fit_and_evaluate(grid: GridSearchCV,
                     X_train, y_train,
                     X_val,   y_val,
                     random_state=None,
                     infer_repeats: int = 3,
                     per_sample_measure: int = 100) -> Tuple[GridSearchCV, Dict[str, Any]]:

    # ---------- training time ----------
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)  #vec__random_state=random_state, clf__random_state=random_state
    t1 = time.perf_counter() - t0
    train_time = time.perf_counter() - t0

    best_pipe = grid.best_estimator_

    # ---------- warm-up (lazy model loading, embeddings etc.) ----------
    try:
        _ = best_pipe.predict(list(X_val[:min(5, len(X_val))]))
    except Exception:
        pass

    # ---------- inference time (batch predict) ----------
    median_infer_time, all_times = _measure_fn(lambda: best_pipe.predict(X_val), repeats=infer_repeats)
    infer_total = median_infer_time
    infer_per_sample = infer_total / max(1, len(X_val))
    throughput = len(X_val) / infer_total if infer_total > 0 else None

    # ---------- predict_proba / decision_function timing & auc ----------
    try:
        # warm-up
        _ = best_pipe.predict_proba(list(X_val[:min(5, len(X_val))]))
        prob_time, _ = _measure_fn(lambda: best_pipe.predict_proba(X_val), repeats=infer_repeats)
        probs = best_pipe.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, probs))
    except Exception:
        # если нет predict_proba, попробуем decision_function
        try:
            _ = best_pipe.decision_function(list(X_val[:min(5, len(X_val))]))
            prob_time, _ = _measure_fn(lambda: best_pipe.decision_function(X_val), repeats=infer_repeats)
            scores = best_pipe.decision_function(X_val)
            auc = float(roc_auc_score(y_val, scores))
        except Exception:
            prob_time = None
            auc = None

    # ---------- predictions & report ----------
    y_pred = best_pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    rep = classification_report(y_val, y_pred, output_dict=True)

    # ---------- architecture description ----------
    arch = dump_pipeline_architecture(best_pipe)

    # ---------- model filesize ----------
    try:
        model_size = _model_filesize_bytes(best_pipe)
    except Exception:
        model_size = None

    return grid, {
        "accuracy": float(acc),
        "auc": auc,
        "report": rep,
        "best_arch": arch,
        # timing metrics
        "timing": {
            "train_time_sec": float(train_time),
            "inference": {
                "batch_median_sec": float(infer_total),
                "per_sample_avg_sec": float(infer_per_sample),
                "throughput_samples_per_sec": float(throughput) if throughput is not None else None,
                "probability_time_sec": float(prob_time) if prob_time is not None else None,
            }
        },
        "model_size_bytes": model_size,
    }


def save_model(model, filename="best_model.joblib"):
    with open(MODEL_DIR / filename, "wb") as f:
        joblib.dump(model, f)


def load_model(filename="best_model.joblib"):
    with open(MODEL_DIR / filename, "rb") as f:
        return joblib.load(f)
