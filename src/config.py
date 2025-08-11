from pathlib import Path


from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple, Optional
import yaml

DEFAULT_VECTORIZER = "tfidf"
DEFAULT_CLASSIFIER = "MultinomialNB"


@dataclass
class ExperimentConfig:
    # ---------- DATA ----------
    datasets: List[str]
    sample_size: float = 1
    test_size: float = 0.1

    # ---------- VECTORIZER ----------
    vectorizer: str = DEFAULT_VECTORIZER

    # ---------- CLASSIFIER ----------
    classifier: str = DEFAULT_CLASSIFIER

    # ---------- CV & METRICS ----------
    cv_folds: int = 3
    scoring: str = "f1_macro"
    random_state: int = 42

    # ---------- SPECIAL PARAMS ----------
    vectorizer_params: Dict[str, Any] = field(default_factory=dict)
    classifier_params: Dict[str, Any] = field(default_factory=dict)

    # ---------- OPTIONAL RUN METADATA ----------
    cache_dir: Optional[str] = "data/cache"
    experiment_name: Optional[str] = None  # if not provided, use folder name


def load_experiment_config(path: Path) -> ExperimentConfig:
    with open(path, 'r') as f:
        raw_data = yaml.safe_load(f) or {}

    known_keys = set(ExperimentConfig.__annotations__.keys())
    vec_name = raw_data.get('vectorizer', DEFAULT_VECTORIZER)
    clf_name = raw_data.get('classifier', DEFAULT_CLASSIFIER)

    # Собираем параметры, относящиеся к выбранному векторизатору/классификатору
    vec_params = {k: v for k, v in raw_data.items() if k.startswith(f"{vec_name}__")}
    clf_params = {k: v for k, v in raw_data.items() if k.startswith(f"{clf_name}__")}

    # Преобразования, если нужно
    for k, v in list(vec_params.items()):
        if "range" in k and isinstance(v, list):
            vec_params[k] = [tuple(r) for r in v]

    normal_data = {k: v for k, v in raw_data.items() if k in known_keys}
    normal_data['vectorizer_params'] = vec_params
    normal_data['classifier_params'] = clf_params

    return ExperimentConfig(**normal_data)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return {
        config.vectorizer: config.vectorizer_params,
        config.classifier: config.classifier_params,
    }