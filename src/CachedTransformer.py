# src/CachedTransformer.py (replace CachedTransformer with this version)
import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _hash_iterable_texts(X: Iterable[str], extra: dict | None = None) -> str:
    h = hashlib.sha256()
    sorted_texts = sorted(str(t) for t in X)
    for s in sorted_texts:
        if s is None:
            s = ""
        if not isinstance(s, (bytes, bytearray)):
            s = str(s).encode("utf-8", errors="ignore")
        h.update(s)
        h.update(b"\0")
    if extra:
        extra_bytes = json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h.update(b"||EXTRA||")
        h.update(extra_bytes)
    return h.hexdigest()


def _make_json_safe(obj):
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    return obj



class CachedTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper that caches transform(X) result to disk using joblib.

    Important sklearn compat rules:
    - __init__ must only assign constructor args to attributes with the same names
      and must not mutate them (so clone/get_params work).
    - implement get_params / set_params so GridSearchCV can set nested transformer params.
    """
    def __init__(self, transformer, cache_dir: str = "data/cache",
                 prefix: str | None = None, use_cache: bool = True, compress: int = 3):
        # do not convert types here (keep exactly what user passed)
        self.transformer = transformer
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.use_cache = use_cache
        self.compress = compress

    # ---------------- sklearn params API ----------------
    def get_params(self, deep: bool = True):
        # Return constructor params + optionally transformer__*
        params = {
            "transformer": self.transformer,
            "cache_dir": self.cache_dir,
            "prefix": self.prefix,
            "use_cache": self.use_cache,
            "compress": self.compress,
        }
        if deep:
            # expose nested transformer params under transformer__*
            try:
                tparams = self.transformer.get_params(deep=True)
                for k, v in tparams.items():
                    params[f"transformer__{k}"] = v
            except Exception:
                pass
        return params

    def set_params(self, **params):
        """
        Accept both:
         - own params: cache_dir, prefix, use_cache, compress, transformer
         - nested transformer params either as 'transformer__param' OR directly as 'param'
           (the latter is convenient when Pipeline passes 'min_df' to this step).
        """
        own_keys = {"transformer", "cache_dir", "prefix", "use_cache", "compress"}
        transformer_updates = {}

        for k, v in params.items():
            if k in own_keys:
                setattr(self, k, v)
            elif k.startswith("transformer__"):
                transformer_updates[k[len("transformer__"):]] = v
            else:
                # forward unknown top-level names to transformer (e.g. min_df)
                transformer_updates[k] = v

        if transformer_updates:
            if hasattr(self.transformer, "set_params"):
                self.transformer.set_params(**transformer_updates)
            else:
                # fallback: set as attributes directly
                for tk, tv in transformer_updates.items():
                    setattr(self.transformer, tk, tv)
        return self

    # ---------------- behaviour ----------------
    def _get_cache_dir_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def fit(self, X, y=None):
        # forward fit to underlying transformer if it needs fitting
        if hasattr(self.transformer, "fit"):
            self.transformer.fit(X, y)
        return self

    def _cache_path_for(self, X):
        try:
            iter(X)
        except TypeError:
            X = list(X)
        params = {}
        if hasattr(self.transformer, "get_params"):
            try:
                params = self.transformer.get_params(deep=False)
                params = _make_json_safe(params)
            except Exception:
                params = {}
        h = _hash_iterable_texts(X, extra={"cls": self.transformer.__class__.__name__, "params": params})
        fname = f"{(self.prefix or self.transformer.__class__.__name__)}_{h}.joblib"
        return self._get_cache_dir_path() / fname

    def transform(self, X):
        cache_path = self._cache_path_for(X)
        if self.use_cache and cache_path.exists():
            return joblib.load(cache_path)
        arr = self.transformer.transform(X)
        if self.use_cache:
            joblib.dump(arr, cache_path, compress=self.compress)
        return arr

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)
