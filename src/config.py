from dataclasses import dataclass, field
from typing import List, Union, Tuple

import yaml
from pathlib import Path

@dataclass
class PipelineConfig:
    # ---------- DATA ----------
    datasets: List[str]
    test_size: float = 0.1

    # ---------- VECTORIZER ----------
    vectorizer: List[str] = field(default_factory=lambda: ['tfidf'])        # "count" | "tfidf"

    analyzer: List[str] = field(default_factory=lambda: ['word'])      # "word", "char", "char_wb"

    # Search space for vectorizer
    ngram_range: List[Tuple[int, int]] = field(default_factory=lambda: [(1, 2)])
    min_df: List[Union[int, float]] = field(default_factory=lambda: [3])
    max_df: List[Union[int, float]] = field(default_factory=lambda: [0.4])

    # ---------- CLASSIFIER ----------
    alpha: List[float] = field(default_factory=lambda: [0.01])  # MultinomialNB smoothing

    # ---------- CV & METRICS ----------
    cv_folds: int = 5
    scoring: str = "f1_macro"
    random_state: int = 42


def load_yaml(path: Path | str) -> PipelineConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
