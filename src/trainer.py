import pickle
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def fit_and_evaluate(grid: GridSearchCV,
                     X_train, y_train,
                     X_val,   y_val):
    # Fit on all training data (the grid already did this internally)
    grid.fit(X_train, y_train)

    # ---------- predictions ----------
    y_pred = grid.best_estimator_.predict(X_val)

    # ---------- accuracy & report ----------
    acc = accuracy_score(y_val, y_pred)
    rep = classification_report(y_val, y_pred, output_dict=True)

    # ---------- AUC ----------
    # 1) use predict_proba if available
    try:
        probs = grid.best_estimator_.predict_proba(X_val)[:, 1]
    except AttributeError:
        # fallback to decision_function (e.g., LinearSVC)
        probs = grid.best_estimator_.decision_function(X_val)

    auc = roc_auc_score(y_val, probs)

    return {
        "accuracy": acc,
        "auc": auc,
        "report": rep
    }


def save_model(model, filename="best_model.pkl"):
    with open(MODEL_DIR / filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename="best_model.pkl"):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)
