from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SBERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        # Нет обучения — просто возвращаем self
        return self

    def transform(self, X):
        self._ensure_model()
        # model.encode может принимать списки; возвращает numpy array
        return np.array(self._model.encode(list(X), batch_size=self.batch_size, show_progress_bar=False))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)