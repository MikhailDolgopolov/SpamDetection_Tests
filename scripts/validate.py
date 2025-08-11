import json

from src.model_builder import build_pipeline_from_params
from src.utils import split_params

if __name__=="__main__":
    best_params = json.load(open("experiments/my_experiment/best_params.json", "r", encoding="utf-8"))
    vec_params, clf_params = split_params(best_params)
    pipe = build_pipeline_from_params(vec_name=cfg.vectorizer, clf_name=cfg.classifier,
                                      vec_params=vec_params, clf_params=clf_params)