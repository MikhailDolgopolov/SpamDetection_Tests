from pathlib import Path


from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple
import yaml

DEFAULT_VECTORIZER = "tfidf"
DEFAULT_CLASSIFIER = "MultinomialNB"


@dataclass
class ExperimentConfig:
    # ---------- DATA ----------
    datasets: List[str]
    test_size: float = 0.1

    # ---------- VECTORIZER ----------
    vectorizer: str = DEFAULT_VECTORIZER

    # ---------- CLASSIFIER ----------
    classifier: str = DEFAULT_CLASSIFIER

    # ---------- CV & METRICS ----------
    cv_folds: int = 5
    scoring: str = "f1_macro"
    random_state: int = 42

    # ---------- SPECIAL PARAMS ----------
    vectorizer_params: Dict[str, Any] = field(default_factory=dict)
    classifier_params: Dict[str, Any] = field(default_factory=dict)


def load_experiment_config(path: Path) -> ExperimentConfig:
    with open(path, 'r') as f:
        raw_data = yaml.safe_load(f)

    known_keys = set(ExperimentConfig.__annotations__.keys())
    vec_name = raw_data.get('vectorizer', DEFAULT_VECTORIZER)
    clf_name = raw_data.get('classifier', DEFAULT_CLASSIFIER)
    vec_params = {k: v for k, v in raw_data.items() if f"{vec_name}__" in k}
    clf_params = {k: v for k, v in raw_data.items() if f"{clf_name}__" in k}

    for k, v in vec_params.items():
        if "range" in k:
            vec_params[k] = [tuple(r) for r in v]

    for k, v in clf_params.items():
        pass

    normal_data = {k: v for k, v in raw_data.items() if k in known_keys}
    normal_data['vectorizer_params'] = vec_params
    normal_data['classifier_params'] = clf_params

    return ExperimentConfig(**normal_data)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return {
        config.vectorizer: config.vectorizer_params,
        config.classifier: config.classifier_params,
    }