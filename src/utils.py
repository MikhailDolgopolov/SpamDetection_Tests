import json
import platform
import tempfile
import time

import joblib
import numpy as np
import sklearn
import yaml
from pathlib import Path
from typing import Dict, Any, Mapping, Sequence, Optional

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from src.config import ExperimentConfig


def convert_ndarray_to_list(o):
    """Recurse through a dict/list and replace np.ndarray with list."""
    if isinstance(o, Mapping):  # dict-like
        return {k: convert_ndarray_to_list(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        return [convert_ndarray_to_list(i) for i in o]
    elif isinstance(o, sklearn.base.BaseEstimator):
        return type(o).__name__
    else:
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()  # ← вот где конвертируем
        except ImportError:
            pass
        return o


def make_json_safe(obj):
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(o) for o in obj]
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    return obj


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def save_experiment_artifacts(experiment: ExperimentConfig,
                              best_estimator,
                              results: Dict[str, Any],
                              save_yaml_results: bool = True):
    experiment.own_dir.mkdir(parents=True, exist_ok=True)

    results = convert_ndarray_to_list(results)

    with open(experiment.own_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if save_yaml_results:
        with open(experiment.own_dir / "results.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, sort_keys=False, allow_unicode=True)


def split_params(best_params: Dict[str, Any]):
    vec_params = {}
    clf_params = {}
    for k, v in best_params.items():
        if k.startswith("vec__"):
            vec_params[k.replace("vec__", "")] = v
        elif k.startswith("clf__"):
            clf_params[k.replace("clf__", "")] = v

    return vec_params, clf_params


def _model_filesize_bytes(model) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        tmp_path = Path(tmp.name)
    try:
        joblib.dump(model, tmp_path)
        size = tmp_path.stat().st_size
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return int(size)


def _measure_fn(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), [float(t) for t in times]


def measure_inference_metrics(
    pipeline,
    X,
    y,
    sample_size=1000,
    per_sample_measure=100,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    n_total = len(X)
    n_sample = min(sample_size, n_total)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    X_sample = [X[i] for i in idx]
    y_sample = [y[i] for i in idx] if y is not None else None

    # batch predict timing
    t0 = time.perf_counter()
    pipeline.predict(X_sample)
    batch_time = time.perf_counter() - t0
    throughput = n_sample / batch_time if batch_time > 0 else None

    # per-sample latency
    m = min(per_sample_measure, n_sample)
    per_idx = rng.choice(n_sample, size=m, replace=False)
    latencies = []
    for j in per_idx:
        t0 = time.perf_counter()
        pipeline.predict([X_sample[j]])
        latencies.append(time.perf_counter() - t0)
    per_sample_ms = float(np.median(latencies) * 1000)


    auc = None
    y_pred = pipeline.predict(X_sample)
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_sample)[:, 1]
            auc = float(roc_auc_score(y_sample, proba))
    except Exception:
        pass

    return {
        "n_total": n_total,
        "n_sample": n_sample,
        "batch_time_sec": batch_time,
        "auc": auc,
        "throughput_sps": throughput,
        "per_sample_median_ms": per_sample_ms,
        "classification_report": classification_report(y_sample, y_pred, output_dict=True),

    }
