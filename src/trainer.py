import pickle
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from src.model_builder import dump_pipeline_architecture

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def fit_and_evaluate(grid: GridSearchCV,
                     X_train, y_train,
                     X_val,   y_val) -> Dict[str, Any]:

    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_

    # ---------- predictions ----------
    y_pred = best_pipe.predict(X_val)

    # ---------- accuracy & report ----------
    acc = accuracy_score(y_val, y_pred)
    rep = classification_report(y_val, y_pred, output_dict=True)

    # ---------- AUC ----------
    try:
        probs = best_pipe.predict_proba(X_val)[:, 1]
    except AttributeError:          # e.g. LinearSVC
        probs = best_pipe.decision_function(X_val)
    auc = roc_auc_score(y_val, probs)

    # ---------- architecture ----------
    arch = dump_pipeline_architecture(best_pipe)

    return {
        "accuracy": acc,
        "auc": auc,
        "report": rep,
        "best_params": grid.best_params_,
        "architecture": arch,
    }


def save_model(model, filename="best_model.pkl"):
    with open(MODEL_DIR / filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename="best_model.pkl"):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)
