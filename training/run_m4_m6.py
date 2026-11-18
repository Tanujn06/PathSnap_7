"""run_m4_m6.py - pass-1 runs for M4-M6 (TRAIN-05).

Owner: Sinchana. For each model:
  1. 5-fold GroupKFold CV on the 80 train trips (capped rows per fold)
  2. fit on capped train rows, score the training rows (overfitting check)
  3. score the 10 student_test trips, save p_match for the HMM
Appends one row per step to results/runs_m4_m6.csv.

Run:  PYTHONPATH=src:. python training/run_m4_m6.py
      PYTHONPATH=src:. python training/run_m4_m6.py --models knn   # one model
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from training.harness import (SEED, append_run, balanced_row_sample, fit_cv,
                              load_features, make_pipeline, save_probabilities,
                              score_binary)
from training.models_m4_m6 import build_models

CACHE = Path("data_cache")
FEATURE_COLS = json.load(open("configs/feature_columns.json"))["feature_columns"]


def top1_accuracy(df, probs):
    """Per GPS point, pick the candidate with the highest p_match; correct if
    its label is 1. Points whose true arc is not among the candidates count
    as wrong."""
    d = df[["trip_id", "obs_idx", "label"]].reset_index(drop=True)
    d["p"] = probs
    best = d.loc[d.groupby(["trip_id", "obs_idx"])["p"].idxmax()]
    return float(best["label"].mean())


def make_row(name, spec, stage, row_cap, **metrics):
    """Same columns in every row so the CSV header stays valid."""
    row = {"model": name, "family": spec["family"], "stage": stage,
           "fold": "", "seed": SEED, "row_cap": row_cap,
           "train_rows": "", "valid_rows": "", "train_time_s": "",
           "inference_time_s": "", "inference_time_per_trip_s": "",
           "macro_f1": "", "log_loss": "", "top1": "",
           "config": json.dumps(spec["params"])}
    row.update(metrics)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["knn", "hist_gradient_boosting", "mlp"])
    ap.add_argument("--quick", action="store_true", help="tiny row cap, for testing only")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = Path(args.results)
    runs_csv = out / "runs_m4_m6.csv"

    train = load_features(CACHE / "features_train.parquet")
    test = load_features(CACHE / "features_student_test.parquet")
    print(f"train {len(train):,} rows / {train.trip_id.nunique()} trips, "
          f"student_test {len(test):,} rows / {test.trip_id.nunique()} trips")

    for name in args.models:
        spec = build_models()[name]
        cap = 5_000 if args.quick else spec["row_cap"]
        print(f"\n=== {name} (row cap {cap:,}) ===")

        # 1. Group-aware cross-validation on the 80 train trips
        t0 = time.perf_counter()
        folds = fit_cv(spec["model"], train, feature_cols=FEATURE_COLS,
                       scale=spec["scale"], row_cap=cap)
        for f in folds:
            append_run(runs_csv, make_row(name, spec, "cv", cap, **f))
            print(f"  fold {f['fold']}: macro_f1={f['macro_f1']:.4f} log_loss={f['log_loss']:.4f}")
        print(f"  CV took {time.perf_counter() - t0:.1f}s")

        # 2. Fit once on capped train rows
        tr = balanced_row_sample(train, cap, SEED)
        pipe = make_pipeline(spec["model"], scale=spec["scale"])
        t0 = time.perf_counter()
        pipe.fit(tr[FEATURE_COLS], tr["label"])
        train_s = time.perf_counter() - t0

        # 2b. Score on its own training rows (compare with CV for overfitting)
        p_tr = pipe.predict_proba(tr[FEATURE_COLS])[:, 1]
        append_run(runs_csv, make_row(name, spec, "train_fit", cap,
                                      train_rows=len(tr), train_time_s=train_s,
                                      **score_binary(tr["label"], p_tr)))

        # 3. Student-test evaluation
        t0 = time.perf_counter()
        p_te = pipe.predict_proba(test[FEATURE_COLS])[:, 1]
        infer_s = time.perf_counter() - t0
        scores = score_binary(test["label"], p_te)
        top1 = top1_accuracy(test, p_te)
        append_run(runs_csv, make_row(
            name, spec, "student_test", cap, train_rows=len(tr), valid_rows=len(test),
            train_time_s=train_s, inference_time_s=infer_s,
            inference_time_per_trip_s=infer_s / test.trip_id.nunique(),
            top1=top1, **scores))
        print(f"  student_test: macro_f1={scores['macro_f1']:.4f} "
              f"log_loss={scores['log_loss']:.4f} top1={top1:.4f}")

        # Probabilities for the HMM (Simon) - data_cache is gitignored
        save_probabilities(CACHE / f"probs_{name}_student_test.parquet", test, p_te)

        # MLP is the only model with an epoch-wise training history
        if name == "mlp":
            m = pipe.named_steps["model"]
            pd.DataFrame({"epoch": range(1, len(m.loss_curve_) + 1),
                          "train_loss": m.loss_curve_,
                          "val_accuracy": m.validation_scores_}
                         ).to_csv(out / "mlp_history.csv", index=False)
            print(f"  MLP history: {len(m.loss_curve_)} epochs -> {out/'mlp_history.csv'}")


if __name__ == "__main__":
    main()
