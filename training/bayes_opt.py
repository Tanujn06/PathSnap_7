"""Optuna tuning for five-fold grouped top-1 accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from optuna.trial import TrialState
from sklearn.ensemble import HistGradientBoostingClassifier

from training.harness import (
    SEED,
    balanced_row_sample,
    group_cv,
    load_features,
    make_pipeline,
    score_binary,
    split_xy,
)
from training.run_m1_m3 import top1_accuracy

FEATURE_PATH = Path("data_cache/features_train.parquet")
FEATURE_CONFIG = Path("configs/feature_columns.json")
STORAGE = "sqlite:///results/optuna_study.db"
STUDY_NAME = "pathsnap_hist_gradient_boosting_top1"
ROW_CAP = 300_000


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
    x, y, groups, _ = split_xy(train, feature_cols)

    top1_scores = []
    macro_f1_scores = []
    log_loss_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        group_cv().split(x, y, groups), start=1
    ):
        fold_train = train.iloc[train_idx]
        fold_valid = train.iloc[valid_idx]
        fold_train = balanced_row_sample(fold_train, ROW_CAP, SEED + fold)

        if fold_train["label"].nunique() < 2:
            raise ValueError(f"Fold {fold} training data has only one class")

        pipeline = make_pipeline(model, scale=False)
        pipeline.fit(fold_train[feature_cols], fold_train["label"].astype(int))

        classes = list(pipeline.named_steps["model"].classes_)
        if 1 not in classes:
            raise ValueError(f"Fold {fold} model did not learn class 1")

        probabilities = pipeline.predict_proba(fold_valid[feature_cols])[
            :, classes.index(1)
        ]

        scores = score_binary(fold_valid["label"].astype(int), probabilities)
        top1_scores.append(top1_accuracy(fold_valid, probabilities))
        macro_f1_scores.append(scores["macro_f1"])
        log_loss_scores.append(scores["log_loss"])

    trial.set_user_attr("fold_top1_scores", top1_scores)
    trial.set_user_attr("mean_macro_f1", sum(macro_f1_scores) / len(macro_f1_scores))
    trial.set_user_attr("mean_log_loss", sum(log_loss_scores) / len(log_loss_scores))

    return sum(top1_scores) / len(top1_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["hist_gradient_boosting"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    if args.trials < 20:
        parser.error("At least 20 total trials are required.")

    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(f"Missing {FEATURE_PATH}")

    train = load_features(FEATURE_PATH)
    if train.trip_id.nunique() < 5:
        raise ValueError("At least 5 training trips are required.")

    feature_cols = json.loads(FEATURE_CONFIG.read_text())["feature_columns"]

    Path("results").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.trials - completed)

    if remaining:
        study.optimize(
            lambda trial: objective(trial, train, feature_cols),
            n_trials=remaining,
            timeout=args.timeout,
            gc_after_trial=True,
        )

    if not any(t.state == TrialState.COMPLETE for t in study.trials):
        raise RuntimeError("Optuna produced no completed trials.")

    best = study.best_trial
    result = {
        "model": args.model,
        "optimization_metric": "mean_grouped_top1_accuracy",
        "best_top1_accuracy": best.value,
        "best_macro_f1": best.user_attrs.get("mean_macro_f1"),
        "best_log_loss": best.user_attrs.get("mean_log_loss"),
        "best_params": best.params,
        "feature_columns": feature_cols,
        "seed": SEED,
        "study_name": STUDY_NAME,
        "completed_trials": sum(
            t.state == TrialState.COMPLETE for t in study.trials
        ),
    }

    Path("results/best_model.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
