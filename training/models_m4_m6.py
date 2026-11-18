"""models_m4_m6.py - M4 KNN, M5 HistGradientBoosting, M6 MLP (TRAIN-04).

Owner: Sinchana. Branch: feature/models-m4-m6.
Initial hyperparameters recorded in docs/MODEL_CONFIGS_M4_M6.md.

Each model is an sklearn Pipeline, so any scaler is fitted inside each CV
fold only (no leakage from validation trips). The task is binary: for each
(GPS point, candidate arc) row, predict p_match = P(candidate is the
reference arc). All three models expose predict_proba, which the HMM needs
as emission scores.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# Pass-1 row caps per fold (sprint plan 8.2). KNN and MLP are slow on CPU.
ROW_CAPS = {
    "M4_KNN": 100_000,
    "M5_HistGB": 300_000,
    "M6_MLP": 100_000,
}

# Initial hyperparameters, kept as plain dicts so they can be logged as JSON
# in every run row and reused by bayes_opt.py.
CONFIGS = {
    "M4_KNN": {
        "n_neighbors": 15,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
        "n_jobs": -1,
    },
    "M5_HistGB": {
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": SEED,
    },
    "M6_MLP": {
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
        "random_state": SEED,
    },
}

FAMILIES = {
    "M4_KNN": "Instance-based",
    "M5_HistGB": "Boosting",
    "M6_MLP": "Neural network",
}


def make_knn(**overrides):
    """M4: distance-weighted KNN. Needs scaling because it uses Euclidean
    distance in feature space (metres and cosines must be comparable)."""
    params = {**CONFIGS["M4_KNN"], **overrides}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(**params)),
    ])


def make_histgb(**overrides):
    """M5: histogram gradient boosting. Tree splits are scale-invariant,
    so no scaler is needed."""
    params = {**CONFIGS["M5_HistGB"], **overrides}
    return Pipeline([
        ("clf", HistGradientBoostingClassifier(**params)),
    ])


def make_mlp(**overrides):
    """M6: shallow MLP. Gradient-based, so inputs are standardised.
    After fitting, pipeline.named_steps['clf'].loss_curve_ holds the
    per-epoch training loss and validation_scores_ the validation accuracy
    (the only model here with an epoch-wise history)."""
    params = {**CONFIGS["M6_MLP"], **overrides}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(**params)),
    ])


BUILDERS = {
    "M4_KNN": make_knn,
    "M5_HistGB": make_histgb,
    "M6_MLP": make_mlp,
}


def get_models():
    """Return {name: dict(build, config, family, row_cap)} for the harness."""
    return {
        name: {
            "build": BUILDERS[name],
            "config": CONFIGS[name],
            "family": FAMILIES[name],
            "row_cap": ROW_CAPS[name],
        }
        for name in BUILDERS
    }


if __name__ == "__main__":
    # Smoke test on synthetic data (no real features needed).
    import time

    from sklearn.datasets import make_classification
    from sklearn.metrics import f1_score, log_loss

    X, y = make_classification(n_samples=5000, n_features=12, weights=[0.85],
                               random_state=SEED)
    for name, spec in get_models().items():
        model = spec["build"]()
        t0 = time.time()
        model.fit(X[:4000], y[:4000])
        p = model.predict_proba(X[4000:])[:, 1]
        print(f"{name:10s} fit {time.time() - t0:5.2f}s  "
              f"macroF1={f1_score(y[4000:], p > 0.5, average='macro'):.3f}  "
              f"logloss={log_loss(y[4000:], p):.3f}")
