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


def _convert_ndarray_to_list(o):
    """Recurse through a dict/list and replace np.ndarray with list."""
    if isinstance(o, Mapping):  # dict-like
        return {k: _convert_ndarray_to_list(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        return [_convert_ndarray_to_list(i) for i in o]
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

def _make_json_safe(obj):
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    return obj

def save_experiment_artifacts(experiment: ExperimentConfig,
                              best_estimator,
                              results: Dict[str, Any],
                              save_yaml_results: bool = True):
    experiment.own_dir.mkdir(parents=True, exist_ok=True)

    results = _convert_ndarray_to_list(results)

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


def measure_and_save_inference_metrics(
        pipeline,
        X: Sequence[str],
        y: Sequence,
        exp_dir: Path,
        pipeline_fname: str = "final/best_pipeline.joblib",
        out_fname: str = "final/inference_metrics.json",
        summary_csv: str = "final/inference_metrics_summary.csv",
        sample_size: int = 10000,
        per_sample_measure: int = 200,
        repeats: int = 3,
        random_state: int = 42,
        save_yaml: bool = True,
) -> Dict[str, Any]:
    """
    Measure inference metrics and save detailed JSON + YAML and append a one-row summary CSV.
    - pipeline: trained pipeline object (or path, but prefer object)
    - X: array-like of texts (can be full dataset)
    - y: optional labels for classification metrics
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    final_dir = exp_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)

    n_total = len(X)
    n_sample = min(sample_size, n_total)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    X_sample = [X[i] for i in idx]
    y_sample = [y[i] for i in idx] if y is not None else None

    # warm-up
    try:
        _ = pipeline.predict(X_sample[:min(5, len(X_sample))])
    except Exception:
        pass

    # batch predict timing
    batch_median_sec, batch_times = _measure_fn(lambda: pipeline.predict(X_sample), repeats=repeats)
    throughput = len(X_sample) / batch_median_sec if batch_median_sec > 0 else None

    # per-sample latency (sampled indices)
    m = min(per_sample_measure, len(X_sample))
    per_idx = rng.choice(len(X_sample), size=m, replace=False)
    latencies = []
    for j in per_idx:
        x = [X_sample[j]]
        t0 = time.perf_counter()
        try:
            pipeline.predict(x)
        except Exception:
            continue
        latencies.append(time.perf_counter() - t0)
    per_sample_median_ms = float(np.median(latencies) * 1000) if latencies else None
    per_sample_p95_ms = float(np.percentile(latencies, 95) * 1000) if latencies else None

    # predict_proba or decision_function timing + auc (if y present)
    prob_time = None
    auc = None
    if y_sample is not None:
        try:
            # warm-up
            _ = pipeline.predict_proba(X_sample[:min(5, len(X_sample))])
            prob_time, _ = _measure_fn(lambda: pipeline.predict_proba(X_sample), repeats=repeats)
            y_pred_proba = pipeline.predict_proba(X_sample)
            if y_sample is not None:
                try:
                    auc = float(roc_auc_score(y_sample, y_pred_proba[:, 1]))
                except Exception:
                    auc = None
        except Exception:
            # fallback to decision_function
            try:
                _ = pipeline.decision_function(X_sample[:min(5, len(X_sample))])
                prob_time, _ = _measure_fn(lambda: pipeline.decision_function(X_sample), repeats=repeats)
                scores = pipeline.decision_function(X_sample)
                try:
                    auc = float(roc_auc_score(y_sample, scores))
                except Exception:
                    auc = None
            except Exception:
                prob_time = None
                auc = None

        # predictions & report
        try:
            y_pred = pipeline.predict(X_sample)
            rep = classification_report(y_sample, y_pred, output_dict=True)
            acc = float(accuracy_score(y_sample, y_pred))
        except Exception:
            rep = None
            acc = None
    else:
        rep = None
        acc = None

    model_size = _model_filesize_bytes(pipeline)

    # model class names & params (top-level)
    try:
        arch = {}
        if hasattr(pipeline, "named_steps"):
            for name, est in pipeline.named_steps.items():
                arch[name] = {
                    "class_name": est.__class__.__name__,
                }
                try:
                    arch[name]["params"] = est.get_params(deep=False)
                except Exception:
                    arch[name]["params"] = {}
    except Exception:
        arch = {}

    # env info
    import sys, sklearn
    env = {
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }

    result = {
        "experiment": exp_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_file": pipeline_fname,
        "model": arch,
        "data": {"n_total": n_total, "n_sample": n_sample, "sample_seed": int(random_state)},
        "metrics": {
            "batch_predict": {"median_sec": float(batch_median_sec), "repeats_sec": batch_times},
            "per_sample": {"median_ms": per_sample_median_ms, "p95_ms": per_sample_p95_ms,
                           "samples_measured": len(latencies)},
            "throughput_samples_per_sec": float(throughput) if throughput is not None else None,
            "predict_proba_sec": float(prob_time) if prob_time is not None else None,
            "auc": float(auc) if auc is not None else None,
            "accuracy": float(acc) if acc is not None else None,
            "classification_report": rep,
        },
        "model_size_bytes": int(model_size),
        "env": env,
    }

    result = _make_json_safe(_convert_ndarray_to_list(result))

    # save json + yaml
    out_path = exp_dir / out_fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if save_yaml:
        with open(out_path.with_suffix(".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)

    # append summary CSV one-line for quick aggregation
    summary_path = exp_dir / summary_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_line = {
        "experiment": exp_dir.name,
        "timestamp": result["timestamp"],
        "model_size_mb": result["model_size_bytes"] / (1024 * 1024),
        "n_sample": n_sample,
        "median_ms": result["metrics"]["per_sample"]["median_ms"],
        "p95_ms": result["metrics"]["per_sample"]["p95_ms"],
        "throughput_sps": result["metrics"]["throughput_samples_per_sec"],
        "auc": result["metrics"]["auc"],
        "accuracy": result["metrics"]["accuracy"],
    }

    # append header if file not exists
    import csv
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_line.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_line)

    return result
