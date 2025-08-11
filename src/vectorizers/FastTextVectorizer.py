import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from sklearn.base import BaseEstimator, TransformerMixin

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """
    Простой `Transformer`, который использует предобученные unsupervised
    FastText‑векторы из файла .vec.
    """

    def __init__(self,
                 model_path: str = "crawl-300d-2M.vec/crawl-300d-2M.vec",
                 min_freq: int = 1,   # минимальное число появлений в корпусе
                 ):
        self.model_path = model_path
        self.min_freq = min_freq
        self._model = None          # будет хранить gensim‑объект

    def fit(self, X, y=None):
        """Загружаем модель один раз."""
        if self._model is None:
            print("Loading FastText vectors …")
            self._model = load_facebook_vectors(self.model_path)
            print(f"Loaded {len(self._model.wv)} word vectors.")
        return self

    def transform(self, X):
        """Векторизует список строк."""
        # Проверяем, что модель загружена
        if self._model is None:
            raise RuntimeError("Call fit() before calling transform().")

        vecs = [self._vectorize(text) for text in X]
        return np.vstack(vecs)

    def _vectorize(self, text: str) -> np.ndarray:
        """Векторизует одну строку."""
        # Разбиваем на слова и отбрасываем короткие токены
        words = [w for w in text.lower().split() if len(w) > 2]

        # Фильтруем по частоте (если есть такой атрибут)
        if hasattr(self._model.wv, "get_vecattr"):
            words = [
                w for w in words
                if self._model.wv.get_vecattr(w, "count") >= self.min_freq
            ]

        # Берём только слова, которые есть в словаре модели
        vecs = [self._model.wv[w] for w in words if w in self._model.wv]

        return np.mean(vecs, axis=0) if vecs else np.zeros(self._model.vector_size)

