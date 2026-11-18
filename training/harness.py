"""Common training helpers for PathSnap (TRAIN-01)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

SEED = 42
N_SPLITS = 5
LABEL_COL = "label"
GROUP_COL = "trip_id"
ID_COLS = ["trip_id", "obs_idx", "arc_id"]

# These are the limits from the sprint plan. The final refit is not capped.
ROW_CAPS = {
    "logistic_regression": 300_000,
    "gaussian_nb": 300_000,
    "hist_gradient_boosting": 300_000,
    "random_forest": 100_000,
    "knn": 100_000,
    "mlp": 100_000,
}


def load_features(path: str | Path) -> pd.DataFrame:
    """Load a feature parquet and check the columns the team agreed on."""
    df = pd.read_parquet(path)
    required = {*ID_COLS, LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Feature table is empty")
    if df[LABEL_COL].isna().any() or df[GROUP_COL].isna().any():
        raise ValueError("label and trip_id cannot contain null values")
    if not set(df[LABEL_COL].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0/1 values")
    return df


def split_xy(df: pd.DataFrame, feature_cols: list[str] | None = None):
    """Keep IDs out of X; use the fixed feature list when the caller has it."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (*ID_COLS, LABEL_COL)]
    if not feature_cols:
        raise ValueError("No feature columns found")
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    x = df[feature_cols].apply(pd.to_numeric, errors="raise")
    x = x.replace([np.inf, -np.inf], np.nan)
    # Tree and linear models cannot handle NaN consistently; use training-fold
    # imputation in the model pipeline instead of filling from the full table.
    y = df[LABEL_COL].astype(int)
    groups = df[GROUP_COL]
    return x, y, groups, feature_cols


def make_pipeline(model, scale: bool = False) -> Pipeline:
    """Wrap a model so any preprocessing is fitted inside each CV fold."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def row_sample(df: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> pd.DataFrame:
    """Cap a training fold while keeping the sample spread across trips."""
    if max_rows is None or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    trips = list(df[GROUP_COL].drop_duplicates())
    per_trip = max(1, max_rows // len(trips))
    picks = []
    for trip in trips:
        part = df[df[GROUP_COL] == trip]
        picks.extend(rng.choice(part.index.to_numpy(),
                                size=min(len(part), per_trip), replace=False))
    picks = np.asarray(picks, dtype=int)
    if len(picks) > max_rows:
        picks = rng.choice(picks, size=max_rows, replace=False)
    elif len(picks) < max_rows:
        left = df.drop(index=picks)
        extra = rng.choice(left.index.to_numpy(),
                           size=min(max_rows - len(picks), len(left)), replace=False)
        picks = np.concatenate([picks, extra])
    return df.loc[picks].sort_index()


def balanced_row_sample(df: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> pd.DataFrame:
    """Alias kept for the model runners; samples across trips."""
    return row_sample(df, max_rows, seed)


def group_cv(n_splits: int = N_SPLITS) -> GroupKFold:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    return GroupKFold(n_splits=n_splits)


def score_binary(y_true, probs) -> dict[str, float]:
    """Candidate-level metrics used for the initial model comparison."""
    probs = np.asarray(probs, dtype=float).reshape(-1)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    pred = (probs >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - probs, probs]), labels=[0, 1])),
    }


def fit_cv(model, df: pd.DataFrame, feature_cols: list[str] | None = None,
           scale: bool = False, row_cap: int | None = None,
           seed: int = SEED) -> list[dict[str, Any]]:
    """Run group-aware CV and return one record per fold."""
    x, y, groups, cols = split_xy(df, feature_cols)
    if groups.nunique() < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} trips for GroupKFold; got {groups.nunique()}")

    out = []
    for fold, (tr_idx, va_idx) in enumerate(group_cv().split(x, y, groups), start=1):
        tr = df.iloc[tr_idx]
        tr = balanced_row_sample(tr, row_cap, seed + fold)
        va = df.iloc[va_idx]
        pipe = make_pipeline(model, scale=scale)
        t0 = time.perf_counter()
        if tr[LABEL_COL].nunique() < 2:
            raise ValueError(f"Fold {fold} training split has only one class")
        pipe.fit(tr[cols], tr[LABEL_COL].astype(int))
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        if not hasattr(pipe, "predict_proba"):
            raise TypeError("Model must support predict_proba for HMM emissions")
        prob = pipe.predict_proba(va[cols])[:, 1]
        infer_s = time.perf_counter() - t0
        scores = score_binary(va[LABEL_COL].astype(int), prob)
        out.append({
            "fold": fold,
            "train_rows": len(tr),
            "valid_rows": len(va),
            "train_time_s": train_s,
            "inference_time_s": infer_s,
            **scores,
        })
    return out


def append_run(path: str | Path, row: dict[str, Any]) -> None:
    """Append one result row; different model runners can use separate CSVs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_probabilities(path: str | Path, frame: pd.DataFrame,
                       probs: Iterable[float]) -> None:
    """Save IDs and p(match) for HMM / pipeline work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[ID_COLS].copy()
    out["p_match"] = np.asarray(list(probs), dtype=float)
    out.to_parquet(path, index=False)
