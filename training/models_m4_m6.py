"""M4-M6 model definitions for PathSnap (TRAIN-04).

Owner: Sinchana. Same interface as training/models_m1_m3.py so the shared
harness (training/harness.py) runs all six models the same way:

    spec = build_models()["knn"]
    fit_cv(spec["model"], df, scale=spec["scale"], row_cap=spec["row_cap"])

The harness wraps every model in a Pipeline (median imputation, then
StandardScaler when scale=True), so preprocessing is fitted inside each
training fold only. Initial hyperparameters: docs/MODEL_CONFIGS_M4_M6.md.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from training.harness import ROW_CAPS, SEED


# Keep the model names stable: they match harness.ROW_CAPS and are used in
# the run CSV and the probs_<model>_{split}.parquet file names.
def build_models(seed: int = SEED) -> dict[str, dict]:
    """Return M4-M6 and the settings the runner needs.

    KNN and MLP use distances / gradient steps, so they are scaled.
    HistGradientBoosting splits on thresholds, so it needs no scaling.
    All three expose predict_proba for the HMM emission scores.
    """
    knn_params = {
        "n_neighbors": 15,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
        "n_jobs": -1,
    }
    hgb_params = {
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": seed,
    }
    mlp_params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 256,
        "learning_rate_init": 1e-3,
        "max_iter": 100,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": seed,
    }
    return {
        "knn": {
            "model": KNeighborsClassifier(**knn_params),
            "family": "Instance-based",
            "scale": True,
            "row_cap": ROW_CAPS["knn"],
            "params": knn_params,
        },
        "hist_gradient_boosting": {
            "model": HistGradientBoostingClassifier(**hgb_params),
            "family": "Boosting",
            "scale": False,
            "row_cap": ROW_CAPS["hist_gradient_boosting"],
            "params": hgb_params,
        },
        "mlp": {
            # Only model with an epoch-wise history: after fitting,
            # pipe.named_steps["model"].loss_curve_ and .validation_scores_
            "model": MLPClassifier(**mlp_params),
            "family": "Neural network",
            "scale": True,
            "row_cap": ROW_CAPS["mlp"],
            "params": mlp_params,
        },
    }


def get_model(name: str, seed: int = SEED):
    """Get one model spec by its stable key."""
    models = build_models(seed)
    if name not in models:
        raise KeyError(f"Unknown model {name!r}. Choose from: {', '.join(models)}")
    return models[name]
