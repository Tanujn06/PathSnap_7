"""BO-02: refit the tuned model on all training rows and export it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from training.harness import SEED, load_features, make_pipeline

FEATURE_PATH = Path("data_cache/features_train.parquet")
CONFIG_PATH = Path("configs/feature_columns.json")
BEST_PATH = Path("results/best_model.json")
MODEL_PATH = Path("models/final.joblib")
EXPORT_CONFIG = Path("models/config.json")
STORAGE = "sqlite:///results/optuna_study.db"
STUDY_NAME = "pathsnap_hist_gradient_boosting"


def main():
    for path in (FEATURE_PATH, CONFIG_PATH, BEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    best = json.loads(BEST_PATH.read_text())
    if best["model"] != "hist_gradient_boosting":
        raise ValueError("Unsupported selected model in best_model.json")

    train = load_features(FEATURE_PATH)
    feature_cols = json.loads(CONFIG_PATH.read_text())["feature_columns"]

    missing = set(feature_cols) - set(train.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    params = dict(best["best_params"])
    params.update({
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": SEED,
    })

    model = HistGradientBoostingClassifier(**params)
    pipeline = make_pipeline(model, scale=False)
    pipeline.fit(train[feature_cols], train["label"].astype(int))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    config = {
        **best,
        "feature_columns": feature_cols,
        "training_rows": len(train),
        "training_trips": int(train.trip_id.nunique()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(MODEL_PATH),
    }
    EXPORT_CONFIG.write_text(json.dumps(config, indent=2))
    print(f"Exported model: {MODEL_PATH}")
    print(f"Exported config: {EXPORT_CONFIG}")


if __name__ == "__main__":
    main()
