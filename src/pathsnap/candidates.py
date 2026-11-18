"""candidates.py - candidate road segments per GPS point (DATA-02).

Owner: Simon C A. Branch: feature/candidates.

For every GPS observation, up to K=10 candidate road arcs with exact
point-to-segment distance and along-arc fraction, written to
data_cache/cands_{split}.parquet with columns
trip_id, obs_idx, arc_id, dist_m, along_frac, cand_rank (docs/CONTRACT.md).

Method (sprint plan DATA-02):
- Each trip has its OWN road network, so the spatial index is built per trip.
- Arcs are densified with sample points about every 10 m, each tagged with
  its arc id, and indexed in a scipy cKDTree. Indexing midpoints only
  plateaus at 87-99 percent recall because long arcs have midpoints far
  from the GPS point.
- Query radius 75 m; if fewer than 3 unique arcs are found, widen to 150 m.
- Both directions of a two-way road are separate arcs, so both are
  candidate arcs naturally.
- Observations with no arc within 150 m get no rows; the HMM handles
  empty observations (plan 8.3).

Trips run in parallel with multiprocessing; results cache to parquet.
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathsnap.data_io import arc_geometry, load_trip, project, trip_origin

K = 10
RADIUS_M = 75.0
WIDE_RADIUS_M = 150.0
MIN_ARCS = 3
DENSIFY_STEP_M = 10.0

COLUMNS = ["trip_id", "obs_idx", "arc_id", "dist_m", "along_frac", "cand_rank"]


def densify_arcs(
    start_xy: np.ndarray, end_xy: np.ndarray, length: np.ndarray, step: float = DENSIFY_STEP_M
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points along every arc, about every ``step`` metres.

    Returns (sample_xy, sample_arc_id): sample points and the arc each
    sample belongs to.
    """
    n_seg = np.maximum((length / step).astype(np.int64), 1)
    counts = n_seg + 1  # include both endpoints
    total = int(counts.sum())
    sample_xy = np.empty((total, 2), dtype=np.float64)
    sample_arc = np.empty(total, dtype=np.int64)
    pos = 0
    for i in range(len(length)):
        n = counts[i]
        fr = np.linspace(0.0, 1.0, n)
        sample_xy[pos : pos + n, 0] = start_xy[i, 0] + fr * (end_xy[i, 0] - start_xy[i, 0])
        sample_xy[pos : pos + n, 1] = start_xy[i, 1] + fr * (end_xy[i, 1] - start_xy[i, 1])
        sample_arc[pos : pos + n] = i
        pos += n
    return sample_xy, sample_arc


def _point_to_segment(
    px: float, py: float, start_xy: np.ndarray, end_xy: np.ndarray, seg_len2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact distance and along-arc fraction from one point to many arcs."""
    dx = end_xy[:, 0] - start_xy[:, 0]
    dy = end_xy[:, 1] - start_xy[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((px - start_xy[:, 0]) * dx + (py - start_xy[:, 1]) * dy) / seg_len2
    t = np.where(seg_len2 > 0.0, t, 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj_x = start_xy[:, 0] + t * dx
    proj_y = start_xy[:, 1] + t * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t


def candidates_for_trip(trip_dir: str | Path) -> pd.DataFrame:
    """Generate the candidate table for one trip."""
    trip = load_trip(trip_dir)
    lon0, lat0 = trip_origin(trip)
    geom = arc_geometry(trip.nodes, trip.arcs, lon0, lat0)
    start_xy, end_xy, length = geom["start_xy"], geom["end_xy"], geom["length"]
    seg_len2 = length**2

    sample_xy, sample_arc = densify_arcs(start_xy, end_xy, length)
    tree = cKDTree(sample_xy)

    obs_x, obs_y = project(trip.track[:, 0], trip.track[:, 1], lon0, lat0)
    obs_xy = np.column_stack([obs_x, obs_y])

    rows = []
    for obs_idx, (px, py) in enumerate(obs_xy):
        hits = tree.query_ball_point([px, py], r=RADIUS_M)
        arc_ids = np.unique(sample_arc[hits])
        if len(arc_ids) < MIN_ARCS:
            hits = tree.query_ball_point([px, py], r=WIDE_RADIUS_M)
            arc_ids = np.unique(sample_arc[hits])
        if len(arc_ids) == 0:
            continue
        dist, along = _point_to_segment(px, py, start_xy[arc_ids], end_xy[arc_ids], seg_len2[arc_ids])
        order = np.argsort(dist, kind="stable")[:K]
        rows.append(
            pd.DataFrame(
                {
                    "trip_id": trip.trip_id,
                    "obs_idx": obs_idx,
                    "arc_id": arc_ids[order],
                    "dist_m": dist[order],
                    "along_frac": along[order],
                    "cand_rank": np.arange(len(order), dtype=np.int64),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(rows, ignore_index=True)[COLUMNS]


def run_split(data_root: str | Path, split: str, out_dir: str | Path, jobs: int = 1) -> Path:
    """Generate candidates for every trip under a split root and write parquet."""
    from pathsnap.data_io import list_trips

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trip_dirs = list_trips(data_root)
    if jobs > 1 and len(trip_dirs) > 1:
        with mp.Pool(jobs) as pool:
            frames = pool.map(candidates_for_trip, trip_dirs)
    else:
        frames = [candidates_for_trip(d) for d in trip_dirs]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    out_path = out_dir / f"cands_{split}.parquet"
    table.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate arcs per GPS point (DATA-02).")
    parser.add_argument("--data-root", required=True, help="Split root containing trip folders")
    parser.add_argument("--split", required=True, help="Split name, e.g. train or student_test")
    parser.add_argument("--out-dir", default="data_cache", help="Output directory for cands_{split}.parquet")
    parser.add_argument("--jobs", type=int, default=max(mp.cpu_count() - 1, 1))
    args = parser.parse_args()
    out_path = run_split(args.data_root, args.split, args.out_dir, jobs=args.jobs)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
