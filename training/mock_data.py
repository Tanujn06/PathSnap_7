"""Tiny feature table for testing the training code before FEAT-01 is ready."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# Keep the schema close to features_{split}; these are fake rows, not results.
def make_mock(n_trips: int = 10, rows_per_trip: int = 40,
              seed: int = SEED) -> pd.DataFrame:
    if n_trips < 5:
        raise ValueError("Need at least 5 trips for GroupKFold(5)")
    if rows_per_trip < 2:
        raise ValueError("Each trip needs at least 2 rows")

    rng = np.random.default_rng(seed)
    n = n_trips * rows_per_trip
    trip_ids = np.repeat([f"mock_{i:02d}" for i in range(n_trips)], rows_per_trip)
    obs = np.tile(np.arange(rows_per_trip), n_trips)
    # One positive candidate among roughly ten candidates per observation.
    label = (rng.random(n) < 0.25).astype(int)
    dist = np.where(label == 1, rng.normal(7, 2, n), rng.normal(24, 8, n))
    heading = np.where(label == 1, rng.normal(12, 8, n), rng.normal(75, 35, n))
    speed = rng.normal(11, 3, n).clip(0, None)

    return pd.DataFrame({
        "trip_id": trip_ids,
        "obs_idx": obs,
        "arc_id": rng.integers(0, 200, n),
        "label": label,
        "dist_m": dist.clip(0, None),
        "along_frac": rng.random(n),
        "cand_rank": rng.integers(1, 11, n),
        "heading_diff_cos": np.cos(np.deg2rad(heading)),
        "speed_mps": speed,
        "dt_prev_s": rng.choice([1.0, 1.0, 1.0, 2.0, 3.0], n),
        "cand_within_20m": rng.integers(1, 6, n),
        "arc_in_degree": rng.integers(1, 5, n),
        "arc_out_degree": rng.integers(1, 5, n),
        "has_reverse_arc": rng.integers(0, 2, n),
    })


def write_mock(path: str | Path = "data_cache/features_mock.parquet",
               **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_mock(**kwargs).to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    out = write_mock()
    print(f"wrote {out}")
