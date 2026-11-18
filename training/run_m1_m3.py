"""run the first training pass for m1-m3."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from training.harness import (
    LABEL_COL,
    SEED,
    group_cv,
    load_features,
    make_pipeline,
    row_sample,
    save_probabilities,
    split_xy,
)
from training.mock_data import make_mock
from training.models_m1_m3 import build_models


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONFIG = ROOT / "configs" / "feature_columns.json"



def load_feature_columns(
    df: pd.DataFrame,
    allow_fallback: bool = False,
) -> list[str]:
    """use the team's frozen feature order when it is available."""
    if FEATURE_CONFIG.exists():
        config = json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))
        columns = config.get("feature_columns", [])

        if columns:
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"configured features are missing: {sorted(missing)}"
                )
            return columns

    if not allow_fallback:
        raise ValueError(
            "feature_columns.json is missing or empty. "
            "wait for the frozen feature list before running real training."
        )

    warnings.warn(
        "using numeric columns because this is a smoke test; "
        "do not report these results.",
        stacklevel=2,
    )
    return split_xy(df)[3]



def top1_accuracy(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    """check whether the highest-scoring candidate is the correct one."""
    required = {"trip_id", "obs_idx", LABEL_COL}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"top-1 input is missing columns: {sorted(missing)}")

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(frame) != len(probabilities):
        raise ValueError("frame and probability lengths do not match")

    if len(frame) == 0:
        raise ValueError("cannot calculate top-1 accuracy on empty data")

    if not np.isfinite(probabilities).all():
        raise ValueError("top-1 probabilities must be finite")

    if not set(frame[LABEL_COL].unique()).issubset({0, 1}):
        raise ValueError("top-1 labels must contain only 0/1 values")

    if frame[["trip_id", "obs_idx"]].isna().any().any():
        raise ValueError("trip_id and obs_idx cannot contain null values")

    ranked = frame[["trip_id", "obs_idx", LABEL_COL]].copy()
    ranked["p_match"] = probabilities

    best = (
        ranked.sort_values("p_match", ascending=False, kind="stable")
        .drop_duplicates(["trip_id", "obs_idx"])
    )

    return float(best[LABEL_COL].mean())


def run_model(
    name: str,
    spec: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    probability_dir: Path,
    seed: int = SEED,
) -> list[dict]:
    """train one model and collect out-of-fold predictions and metrics."""
    from time import perf_counter

    x, y, groups, _ = split_xy(df, feature_cols)

    if groups.nunique() < 5:
        raise ValueError(
            f"need at least 5 unique trips for group cv; got {groups.nunique()}"
        )

    oof = np.full(len(df), np.nan, dtype=float)
    records = []

    for fold, (train_idx, valid_idx) in enumerate(
        group_cv().split(x, y, groups), start=1
    ):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        sampled_train = row_sample(
            train_df,
            spec["row_cap"],
            seed + fold,
        )

        if sampled_train[LABEL_COL].nunique() < 2:
            raise ValueError(f"{name}: fold {fold} training data has one class")

        pipeline = make_pipeline(spec["model"], scale=spec["scale"])

        started = perf_counter()
        pipeline.fit(
            sampled_train[feature_cols],
            sampled_train[LABEL_COL].astype(int),
        )
        train_time = perf_counter() - started

        started = perf_counter()
        probabilities = pipeline.predict_proba(valid_df[feature_cols])
        inference_time = perf_counter() - started

        classes = list(pipeline.named_steps["model"].classes_)
        if 1 not in classes:
            raise ValueError(f"{name}: fold {fold} model did not learn class 1")

        positive_index = classes.index(1)
        probabilities = probabilities[:, positive_index]
        oof[valid_idx] = probabilities

        from training.harness import score_binary

        scores = score_binary(valid_df[LABEL_COL].astype(int), probabilities)
        scores["top1_accuracy"] = top1_accuracy(valid_df, probabilities)

        record = {
            "model": name,
            "fold": fold,
            "train_rows": len(sampled_train),
            "valid_rows": len(valid_df),
            "train_time_s": round(train_time, 4),
            "inference_time_s": round(inference_time, 4),
            **scores,
        }
        records.append(record)



        print(
            f"{name} | fold {fold}/5 | "
            f"macro f1: {scores['macro_f1']:.4f} | "
            f"log loss: {scores['log_loss']:.4f} | "
            f"top-1: {scores['top1_accuracy']:.4f}"
        )

    if np.isnan(oof).any():
        raise RuntimeError(f"{name}: some rows did not receive an oof prediction")

    save_probabilities(
        probability_dir / f"probs_{name}_train.parquet",
        df,
        oof,
    )

    return records


def run_all(
    df: pd.DataFrame,
    output_dir: Path,
    probability_dir: Path,
    seed: int = SEED,
    allow_feature_fallback: bool = False,
) -> dict[str, list[dict]]:
    """run all three models against the same feature table."""
    feature_cols = load_feature_columns(
        df,
        allow_fallback=allow_feature_fallback,
    )

    if not feature_cols:
        raise ValueError("no feature columns available")

    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    all_records = []

    for name, spec in build_models(seed).items():
        print(f"\nstarting {name}...")
        records = run_model(
            name,
            spec,
            df,
            feature_cols,
            output_dir,
            probability_dir,
            seed,
        )
        results[name] = records
        all_records.extend(records)

    pd.DataFrame(all_records).to_csv(
        output_dir / "runs_m1_m3.csv",
        index=False,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="run m1-m3 training experiments")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data_cache" / "features_train.parquet",
        help="path to the training feature parquet",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run on synthetic data for a smoke test only",
    )
    args = parser.parse_args()

    if args.mock:
        print("using synthetic mock data; do not report these as model results")
        df = make_mock()
        output_dir = ROOT / "results" / "mock"
        probability_dir = ROOT / "data_cache" / "mock"
    else:
        if not args.input.exists():
            raise FileNotFoundError(
                f"training data not found: {args.input}\n"
                "wait for FEAT-01 or run with --mock for a smoke test."
            )
        df = load_features(args.input)
        output_dir = ROOT / "results"
        probability_dir = ROOT / "data_cache"

    run_all(
    df,
    output_dir,
    probability_dir,
    allow_feature_fallback=args.mock,
)


if __name__ == "__main__":
    main()
