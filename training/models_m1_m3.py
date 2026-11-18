"""M1-M3 model definitions for PathSnap (TRAIN-02)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from training.harness import ROW_CAPS, SEED


# Keep the model names stable: the runner and probability files use these keys.
def build_models(seed: int = SEED) -> dict[str, dict]:
    """Return the first three classifiers and the settings the runner needs.

    The harness adds median imputation to every model. Only Logistic
    Regression is scaled; trees and GaussianNB do not need that step.
    All three estimators expose predict_proba for the later HMM stage.
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
            "family": "Linear",
            "scale": True,
            "row_cap": ROW_CAPS["logistic_regression"],
            "params": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": seed,
            },
        },
        "gaussian_nb": {
            "model": GaussianNB(),
            "family": "Probabilistic",
            "scale": False,
            "row_cap": ROW_CAPS["gaussian_nb"],
            "params": {},
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                random_state=seed,
                n_jobs=-1,
            ),
            "family": "Ensemble (bagging)",
            "scale": False,
            "row_cap": ROW_CAPS["random_forest"],
            "params": {
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_leaf": 20,
                "random_state": seed,
                "n_jobs": -1,
            },
        },
    }


def get_model(name: str, seed: int = SEED):
    """Get one model spec by its stable key."""
    models = build_models(seed)
    if name not in models:
        raise KeyError(f"Unknown model {name!r}. Choose from: {', '.join(models)}")
    return models[name]
