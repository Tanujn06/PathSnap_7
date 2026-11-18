from pathlib import Path
import json
import time
import joblib

import numpy as np
import pandas as pd

from training.harness import (
    SEED, load_features, make_pipeline, row_sample,
    score_binary, save_probabilities,
)
from training.models_m1_m3 import build_models
from training.run_m1_m3 import load_feature_columns, top1_accuracy

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models" / "m1_m3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train = load_features(CACHE / "features_train.parquet")
test = load_features(CACHE / "features_student_test.parquet")
features = load_feature_columns(train)

old = pd.read_csv(RESULTS / "runs_m1_m3.csv")
rows = []

# Preserve existing CV results and normalize their schema.
# Older run CSVs may not contain top1_accuracy; recover it from OOF probabilities.
cv_top1_by_model = {}

for _, r in old.iterrows():
    # Summary rows have no fold; retain only actual CV fold rows here.
    if pd.isna(r.get("fold")):
        continue

    spec = build_models()[r["model"]]
    model_name = r["model"]

    if "top1_accuracy" in old.columns:
        cv_top1 = float(r["top1_accuracy"])
    else:
        if model_name not in cv_top1_by_model:
            prob_path = CACHE / f"probs_{model_name}_train.parquet"
            if not prob_path.exists():
                raise FileNotFoundError(
                    f"Cannot recover CV top-1: missing {prob_path}"
                )
            pred = pd.read_parquet(prob_path)
            keys = ["trip_id", "obs_idx", "arc_id"]
            truth = train[keys + ["label"]]
            joined = pred.merge(
                truth, on=keys, how="inner", validate="one_to_one"
            )
            if len(joined) != len(pred):
                raise ValueError(f"OOF label join mismatch for {model_name}")
            cv_top1_by_model[model_name] = top1_accuracy(
                joined, joined["p_match"].to_numpy()
            )
        cv_top1 = cv_top1_by_model[model_name]

    rows.append({
        "model": r["model"],
        "family": spec["family"],
        "stage": "cv",
        "fold": int(r["fold"]),
        "seed": SEED,
        "row_cap": spec["row_cap"],
        "train_rows": int(r["train_rows"]),
        "valid_rows": int(r["valid_rows"]),
        "train_time_s": r["train_time_s"],
        "inference_time_s": r["inference_time_s"],
        "inference_time_per_trip_s": np.nan,
        "macro_f1": r["macro_f1"],
        "log_loss": r["log_loss"],
        "top1": cv_top1,
        "config": json.dumps(spec["params"]),
    })

for name, spec in build_models().items():
    print(f"\nEvaluating {name}...")

    sampled = row_sample(train, spec["row_cap"], SEED)
    pipeline = make_pipeline(spec["model"], scale=spec["scale"])

    start = time.perf_counter()
    pipeline.fit(sampled[features], sampled["label"].astype(int))
    fit_time = time.perf_counter() - start

    # Export the fitted pass-1 model and its configuration.
    joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps({
        "model": name,
        "family": spec["family"],
        "seed": SEED,
        "row_cap": spec["row_cap"],
        "training_rows": len(sampled),
        "feature_columns": features,
        "parameters": spec["params"],
    }, indent=2) + "\\n")

    positive_index = list(pipeline.named_steps["model"].classes_).index(1)

    for stage, frame in [("train_fit", sampled), ("student_test", test)]:
        start = time.perf_counter()
        probs = pipeline.predict_proba(frame[features])[:, positive_index]
        inference_time = time.perf_counter() - start

        scores = score_binary(frame["label"].astype(int), probs)
        save_probabilities(
            CACHE / f"probs_{name}_{stage}.parquet",
            frame,
            probs,
        )
        top1 = top1_accuracy(frame, probs)

        rows.append({
            "model": name,
            "family": spec["family"],
            "stage": stage,
            "fold": np.nan,
            "seed": SEED,
            "row_cap": spec["row_cap"],
            "train_rows": len(sampled),
            "valid_rows": len(frame),
            "train_time_s": fit_time,
            "inference_time_s": inference_time,
            "inference_time_per_trip_s": inference_time / frame["trip_id"].nunique(),
            "macro_f1": scores["macro_f1"],
            "log_loss": scores["log_loss"],
            "top1": top1,
            "config": json.dumps(spec["params"]),
        })

        if stage == "student_test":
            save_probabilities(
                CACHE / f"probs_{name}_student_test.parquet",
                frame,
                probs,
            )

        print(
            f"{name} | {stage} | F1={scores['macro_f1']:.4f} "
            f"| log_loss={scores['log_loss']:.4f} | top1={top1:.4f}"
        )

pd.DataFrame(rows).to_csv(RESULTS / "runs_m1_m3.csv", index=False)
print("\nM1-M3 evaluation complete.")
