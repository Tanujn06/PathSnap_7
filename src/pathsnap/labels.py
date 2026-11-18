"""DATA-04: derive observation labels by monotone reference-route alignment.

Development-only module. Never import this into the prediction pipeline.
Run: PYTHONPATH=src python -m pathsnap.labels --data-root data --out data_cache
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from .data_io import Trip, load_trip, list_trips, trip_origin, project, arc_geometry


def align_route(trip: Trip, max_advance: int = 3, return_positions=False):
    """Minimum total point-to-segment distance, monotone route positions.

    Start and end positions are free: a recording can cover part of a route.
    At each observation the state may stay or advance up to max_advance.
    Ties prefer staying. Route positions distinguish repeated arc occurrences.
    No heading or learned features are used to derive the labels.
    """
    if not isinstance(max_advance, int) or max_advance < 0:
        raise ValueError('max_advance must be a nonnegative integer')
    if trip.route is None or len(trip.route) == 0:
        raise ValueError('reference route is required for development labels')
    route = np.asarray(trip.route)
    if route.ndim != 1 or not np.issubdtype(route.dtype, np.integer):
        raise ValueError('route must be a one-dimensional integer array')
    if np.any(route < 0) or np.any(route >= len(trip.arcs)):
        raise ValueError('reference route contains an invalid arc id')
    if len(trip.track) == 0 or not np.isfinite(trip.track).all() or not np.isfinite(trip.nodes).all():
        raise ValueError('track and nodes must be finite and track nonempty')
    lon0, lat0 = trip_origin(trip)
    x, y = project(trip.track[:, 0], trip.track[:, 1], lon0, lat0)
    points = np.column_stack((x, y))
    geom = arc_geometry(trip.nodes, trip.arcs, lon0, lat0)
    start = geom['start_xy'][route]
    vec = geom['end_xy'][route] - start
    length2 = np.einsum('ij,ij->i', vec, vec)
    denominator = np.where(length2 > 0, length2, 1.)
    n, m = len(points), len(route)
    if max_advance > 255:
        raise ValueError('max_advance must be at most 255')
    back = np.zeros((n, m), dtype=np.uint8)
    previous = None
    # Chunk geometry so peak temporary memory does not scale as n*m*2.
    for offset in range(0, n, 128):
        delta = points[offset:offset+128, None, :] - start[None, :, :]
        fraction = np.clip(np.einsum('cmi,mi->cm', delta, vec) / denominator, 0, 1)
        distance = np.linalg.norm(delta - fraction[:, :, None] * vec, axis=2)
        for local_i, cost in enumerate(distance):
            i = offset + local_i
            if previous is None:
                previous = cost.copy()
                continue
            best = previous.copy()
            for jump in range(1, min(max_advance, m-1)+1):
                better = previous[:-jump] < best[jump:]
                best[jump:][better] = previous[:-jump][better]
                back[i, jump:][better] = jump
            previous = cost + best
    positions = np.empty(n, dtype=np.int64)
    positions[-1] = int(np.argmin(previous))
    for i in range(n-1, 0, -1):
        positions[i-1] = positions[i] - int(back[i, positions[i]])
    d = points - start[positions]
    f = np.clip(np.einsum('ij,ij->i', d, vec[positions]) / denominator[positions], 0, 1)
    residual = np.linalg.norm(d - f[:, None] * vec[positions], axis=1)
    frame = pd.DataFrame({'trip_id': str(trip.trip_id), 'obs_idx': np.arange(n),
                          'ref_arc': route[positions], 'ref_dist_m': residual})
    return (frame, positions) if return_positions else frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', default='data_cache')
    parser.add_argument('--diagnostics', default='results/labels')
    args = parser.parse_args()
    out, diagnostic = Path(args.out), Path(args.diagnostics)
    out.mkdir(parents=True, exist_ok=True)
    diagnostic.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ('train', 'student_test'):
        frames = []
        for folder in list_trips(Path(args.data_root) / split):
            start = time.perf_counter()
            frame, positions = align_route(load_trip(folder), return_positions=True)
            frames.append(frame)
            distances = frame.ref_dist_m.to_numpy()
            rows.append(dict(split=split, trip_id=folder.name, n_obs=len(frame),
                             median_ref_dist_m=float(np.median(distances)),
                             p95_ref_dist_m=float(np.percentile(distances,95)),
                             n_above_50m=int((distances>50).sum()),
                             start_position=int(positions[0]), end_position=int(positions[-1]),
                             runtime_s=time.perf_counter()-start))
            print(split, folder.name, len(frame), rows[-1]['median_ref_dist_m'], flush=True)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(out / f'labels_{split}.parquet', index=False)
        summary = dict(split=split, n_trips=len(frames), n_obs=len(combined),
                       median_ref_dist_m=float(combined.ref_dist_m.median()),
                       p95_ref_dist_m=float(combined.ref_dist_m.quantile(.95)),
                       n_above_50m=int((combined.ref_dist_m>50).sum()))
        (diagnostic / f'{split}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(summary, flush=True)
    pd.DataFrame(rows).to_csv(diagnostic/'alignment_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
