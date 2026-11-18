from pathlib import Path
import json
import time

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

train = load_features(CACHE / "features_train.parquet")
test = load_features(CACHE / "features_student_test.parquet")
features = load_feature_columns(train)

old = pd.read_csv(RESULTS / "runs_m1_m3.csv")
rows = []

# Preserve existing CV results and normalize their schema.
for _, r in old.iterrows():
    spec = build_models()[r["model"]]
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
        "top1": r["top1_accuracy"],
        "config": json.dumps(spec["params"]),
    })

for name, spec in build_models().items():
    print(f"\nEvaluating {name}...")

    sampled = row_sample(train, spec["row_cap"], SEED)
    pipeline = make_pipeline(spec["model"], scale=spec["scale"])

    start = time.perf_counter()
    pipeline.fit(sampled[features], sampled["label"].astype(int))
    fit_time = time.perf_counter() - start

    positive_index = list(pipeline.named_steps["model"].classes_).index(1)

    for stage, frame in [("train_fit", sampled), ("student_test", test)]:
        start = time.perf_counter()
        probs = pipeline.predict_proba(frame[features])[:, positive_index]
        inference_time = time.perf_counter() - start

        scores = score_binary(frame["label"].astype(int), probs)
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
