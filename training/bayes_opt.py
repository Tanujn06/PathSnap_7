"""BO-01: Optuna TPE tuning using five-fold GroupKFold macro-F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from sklearn.ensemble import HistGradientBoostingClassifier

from training.harness import SEED, fit_cv, load_features
from training.models_m4_m6 import build_models

FEATURE_PATH = Path("data_cache/features_train.parquet")
FEATURE_CONFIG = Path("configs/feature_columns.json")
STORAGE = "sqlite:///results/optuna_study.db"
STUDY_NAME = "pathsnap_hist_gradient_boosting"


def suggest_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 350, step=50),
        "max_leaf_nodes": trial.suggest_categorical(
            "max_leaf_nodes", [15, 31, 47, 63]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50, step=10),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-5, 10.0, log=True
        ),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": SEED,
    }


def objective(trial, train, feature_cols):
    params = suggest_params(trial)
    model = HistGradientBoostingClassifier(**params)

    folds = fit_cv(
        model,
        train,
        feature_cols=feature_cols,
        scale=False,
        row_cap=300_000,
        seed=SEED,
    )
    scores = [fold["macro_f1"] for fold in folds]
    trial.set_user_attr("fold_scores", scores)
    trial.set_user_attr("mean_log_loss", sum(
        fold["log_loss"] for fold in folds
    ) / len(folds))
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hist_gradient_boosting"],
                        default="hist_gradient_boosting")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    if args.trials < 20:
        parser.error("At least 20 trials are required.")

    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {FEATURE_PATH}. Restore the real training features first."
        )

    train = load_features(FEATURE_PATH)
    if len(train) < 1 or train.trip_id.nunique() < 5:
        raise ValueError("Real training data with at least 5 trips is required.")

    feature_cols = json.loads(FEATURE_CONFIG.read_text())["feature_columns"]
    if "features_mock" in str(FEATURE_PATH):
        raise ValueError("Mock data cannot be used for Bayesian optimization.")

    Path("results").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, train, feature_cols),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    result = {
        "model": args.model,
        "best_macro_f1": study.best_value,
        "best_params": study.best_params,
        "feature_columns": feature_cols,
        "seed": SEED,
        "study_name": STUDY_NAME,
    }
    Path("results/best_model.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
