
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml

DEFAULT_VECTORIZER = "tfidf"
DEFAULT_CLASSIFIER = "MultinomialNB"


@dataclass
class ExperimentConfig:
    datasets: List[str]
    sample_size: float = 1.0
    test_size: float = 0.1

    vectorizer: str = DEFAULT_VECTORIZER
    classifier: str = DEFAULT_CLASSIFIER

    cv_folds: int = 3
    scoring: str = "f1_macro"
    random_state: int = 42

    # vectorizer_params and classifier_params are dicts param_name -> list_of_values
    vectorizer_params: Dict[str, Any] = field(default_factory=dict)
    classifier_params: Dict[str, Any] = field(default_factory=dict)

    own_dir: Optional[Path] = None
    cache_dir: Optional[str] = "data/cache"


def _parse_component_block(block, default_name):
    """
    block can be:
      - string: 'tfidf' -> (name, params={})
      - dict: { tfidf: {min_df: [...], ngram_range: [[1,1],[1,2]] } }
    Returns (name, params_dict)
    """
    if block is None:
        return default_name, {}
    if isinstance(block, str):
        return block, {}
    if isinstance(block, dict):
        if len(block) == 0:
            return default_name, {}
        # take the first (and expected only) key
        name = next(iter(block.keys()))
        params = block[name] or {}
        # normalize any nested range lists to tuples if necessary
        norm = {}
        for k, v in params.items():
            if isinstance(v, list) and all(isinstance(x, list) and len(x) == 2 for x in v):
                # convert list-of-pairs to list-of-tuples (e.g. ngram_range)
                norm[k] = [tuple(x) for x in v]
            else:
                norm[k] = v
        return name, norm
    raise ValueError("Unsupported component block: must be string or single-key dict.")


def load_experiment_config(path: Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # basic known keys
    base_keys = {"datasets", "sample_size", "test_size",
                 "cv_folds", "scoring", "random_state", "own_dir", "cache_dir"}

    # parse vectorizer block
    vec_block = raw.get("vectorizer", None)
    vec_name, vec_params = _parse_component_block(vec_block, DEFAULT_VECTORIZER)

    clf_block = raw.get("classifier", None)
    clf_name, clf_params = _parse_component_block(clf_block, DEFAULT_CLASSIFIER)

    normal_data = {k: v for k, v in raw.items() if k in base_keys}
    normal_data.setdefault("datasets", raw.get("datasets", []))
    normal_data.setdefault("sample_size", raw.get("sample_size", 1.0))
    normal_data.setdefault("test_size", raw.get("test_size", 0.1))

    cfg = ExperimentConfig(**normal_data)
    cfg.vectorizer = vec_name
    cfg.classifier = clf_name
    cfg.vectorizer_params = vec_params
    cfg.classifier_params = clf_params
    if cfg.own_dir is None:
        cfg.own_dir = path.parent

    if isinstance(cfg.own_dir, Path):
        cfg.own_dir = Path(cfg.own_dir)

    return cfg
