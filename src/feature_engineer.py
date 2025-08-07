from typing import List

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(cfg) -> List[sklearn.base.BaseEstimator]:
    vecs = []
    for vectorizer in cfg.vectorizer:
        if vectorizer == "count":
            vecs.append(CountVectorizer())
        if vectorizer == "tfidf":
            vecs.append(TfidfVectorizer())
    return vecs
