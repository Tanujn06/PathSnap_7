import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from training.harness import load_features

MODEL_PATH = Path("models/final.joblib")
CONFIG_PATH = Path("models/config.json")
TEST_PATH = Path("data_cache/features_student_test.parquet")
RUNS_PATH = Path("results/runs_m4_m6.csv")
OUTPUT_PATH = Path("results/tuned_vs_untuned.csv")

def top1_accuracy(df, probs):
    d = df[["trip_id", "obs_idx", "label"]].copy()
    d["p"] = probs
    best = d.loc[d.groupby(["trip_id", "obs_idx"])["p"].idxmax()]
    return float(best["label"].mean())

def evaluate(name, model, test, features):
    start = time.perf_counter()
    probs = model.predict_proba(test[features])[:, 1]
    inference = time.perf_counter() - start

    y = test["label"].astype(int)
    return {
        "model": name,
        "macro_f1": f1_score(y, probs >= 0.5, average="macro"),
        "log_loss": log_loss(y, np.column_stack([1-probs, probs]), labels=[0, 1]),
        "top1_accuracy": top1_accuracy(test, probs),
        "inference_time_s": inference,
        "inference_time_per_trip_s": inference / test.trip_id.nunique(),
        "test_rows": len(test),
        "test_trips": test.trip_id.nunique(),
    }

def main():
    config = json.loads(CONFIG_PATH.read_text())
    features = config["feature_columns"]
    test = load_features(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    tuned = evaluate("hist_gradient_boosting_tuned", model, test, features)

    runs = pd.read_csv(RUNS_PATH)
    baseline_rows = runs[
        (runs.model == "hist_gradient_boosting") &
        (runs.stage == "student_test")
    ]

    if baseline_rows.empty:
        raise RuntimeError("No untuned HistGB student_test baseline found.")

    baseline_row = baseline_rows.iloc[-1]
    baseline = {
        "model": "hist_gradient_boosting_untuned",
        "macro_f1": baseline_row["macro_f1"],
        "log_loss": baseline_row["log_loss"],
        "top1_accuracy": baseline_row["top1"],
        "inference_time_s": baseline_row["inference_time_s"],
        "inference_time_per_trip_s": baseline_row["inference_time_per_trip_s"],
        "test_rows": baseline_row["valid_rows"],
        "test_trips": test.trip_id.nunique(),
    }

    comparison = pd.DataFrame([baseline, tuned])
    comparison["macro_f1_delta_vs_baseline"] = (
        comparison["macro_f1"] - baseline["macro_f1"]
    )
    comparison["log_loss_delta_vs_baseline"] = (
        comparison["log_loss"] - baseline["log_loss"]
    )
    comparison["top1_delta_vs_baseline"] = (
        comparison["top1_accuracy"] - baseline["top1_accuracy"]
    )
    comparison.to_csv(OUTPUT_PATH, index=False)

    # Append exactly one tuned evaluation row; don't duplicate on reruns.
    runs = runs[
        ~(
            (runs.model == "hist_gradient_boosting_tuned") &
            (runs.stage == "student_test")
        )
    ]

    row = {
        "model": "hist_gradient_boosting_tuned",
        "family": "Boosting",
        "stage": "student_test",
        "fold": "",
        "seed": config["seed"],
        "row_cap": "all_training_rows",
        "train_rows": config["training_rows"],
        "valid_rows": len(test),
        "train_time_s": "",
        "inference_time_s": tuned["inference_time_s"],
        "inference_time_per_trip_s": tuned["inference_time_per_trip_s"],
        "macro_f1": tuned["macro_f1"],
        "log_loss": tuned["log_loss"],
        "top1": tuned["top1_accuracy"],
        "config": json.dumps(config["best_params"]),
    }

    pd.concat([runs, pd.DataFrame([row])], ignore_index=True).to_csv(
        RUNS_PATH, index=False
    )

    print("\n=== BASELINE VS TUNED HISTGB ===")
    print(comparison.to_string(index=False))
    print("\nGenerated:", OUTPUT_PATH)
    print("Updated:", RUNS_PATH)
    print("Tuned model training rows:", config["training_rows"])
    print("Tuned model training trips:", config["training_trips"])

if __name__ == "__main__":
    main()
