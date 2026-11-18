"""PIPE-01: reference-free trip prediction with the handed-over model and HMM.

No API wiring here (API-02 is separate). Evaluation/reference loading belongs
outside this module, after prediction. Project-relative defaults do not depend
on the caller's working directory.
"""
import hashlib
import json
import sklearn
import time
import warnings
from dataclasses import fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.exceptions import InconsistentVersionWarning

from .candidates import (densify_arcs, _point_to_segment, COLUMNS, K,
                         RADIUS_M, WIDE_RADIUS_M, MIN_ARCS)
from .data_io import Trip, trip_origin, project, arc_geometry
from .features import build_features, FEATURE_COLUMNS
from .graph import RoadGraph
from .hmm import HMMParams, decode

ROOT = Path(__file__).resolve().parents[2]


def load_prediction_trip(trip_dir):
    """Read required prediction inputs only. Never inspect a .route file."""
    folder = Path(trip_dir)
    tid = folder.name
    arrays = {}
    for ext, dtype, width in [('track', float, 3), ('nodes', float, 2), ('arcs', float, 2)]:
        path = folder / (tid + '.' + ext)
        arrays[ext] = np.loadtxt(path, dtype=dtype, ndmin=2)
        if arrays[ext].size == 0 or arrays[ext].shape[1] != width or not np.isfinite(arrays[ext]).all():
            raise ValueError(f'invalid {ext} table')
    if (arrays['arcs'] != np.floor(arrays['arcs'])).any():
        raise ValueError('arc node indices must be integers')
    arrays['arcs'] = arrays['arcs'].astype(np.int64)
    for ext in ('track', 'nodes'):
        if (np.abs(arrays[ext][:, 0]) > 180).any() or (np.abs(arrays[ext][:, 1]) > 90).any():
            raise ValueError(f'invalid {ext} coordinates')
    if (arrays['arcs'] < 0).any() or (arrays['arcs'] >= len(arrays['nodes'])).any():
        raise ValueError('arc node index out of range')
    if (np.diff(arrays['track'][:, 2]) < 0).any():
        raise ValueError('timestamps must be nondecreasing')
    return Trip(tid, arrays['track'], arrays['nodes'], arrays['arcs'], None)


def prediction_candidates(trip):
    """Same DATA-02 geometry/radii/rank rules, without its path-based route read."""
    origin = trip_origin(trip)
    geom = arc_geometry(trip.nodes, trip.arcs, *origin)
    points, arc_ids = densify_arcs(geom['start_xy'], geom['end_xy'], geom['length'])
    tree = cKDTree(points)
    x, y = project(trip.track[:, 0], trip.track[:, 1], *origin)
    rows = []
    for obs, xy in enumerate(np.column_stack([x, y])):
        ids = np.unique(arc_ids[tree.query_ball_point(xy, r=RADIUS_M)])
        if len(ids) < MIN_ARCS:
            ids = np.unique(arc_ids[tree.query_ball_point(xy, r=WIDE_RADIUS_M)])
        if not len(ids):
            continue
        dist, frac = _point_to_segment(*xy, geom['start_xy'][ids], geom['end_xy'][ids], geom['length'][ids]**2)
        order = np.argsort(dist, kind='stable')[:K]
        rows.extend((trip.trip_id, obs, int(ids[j]), float(dist[j]), float(frac[j]), rank)
                    for rank, j in enumerate(order))
    return pd.DataFrame(rows, columns=COLUMNS)


class Predictor:
    """Load trusted project model/config once; reuse for multiple trips.

    A version mismatch fails by default. An explicit override is diagnostic
    only and is recorded in the returned warnings; never hides the mismatch.
    """
    def __init__(self, model_path=None, config_path=None, hmm_path=None, allow_version_mismatch=False):
        self.provenance = None
        if model_path is None:
            self.provenance = json.loads((ROOT/'models/runtime.json').read_text())
            self.model_path = ROOT/self.provenance['artifact']
            for path, expected in [(self.model_path, self.provenance['sha256']),
                                   (ROOT/self.provenance['original_artifact'], self.provenance['original_sha256'])]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('model provenance hash mismatch')
            if sklearn.__version__ != self.provenance['tested_sklearn']:
                raise RuntimeError('Selected resave is validated only in sklearn ' + self.provenance['tested_sklearn'])
        else:
            self.model_path = Path(model_path)
        config = json.loads(Path(config_path or ROOT/'models/config.json').read_text())
        self.feature_columns = config['feature_columns']
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError('model feature order does not match FEAT-01')
        raw = json.loads(Path(hmm_path or ROOT/'configs/hmm_params.json').read_text())
        names = {f.name for f in fields(HMMParams)}
        if not names.issubset(raw):
            raise ValueError('HMM configuration is incomplete')
        self.params = HMMParams(**{k: raw[k] for k in names})
        vals = np.array([getattr(self.params, k) for k in names], dtype=float)
        if not np.isfinite(vals).all() or self.params.sigma_m <= 0 or self.params.gap_threshold_s < 0 or min(self.params.lambda_per_hop,self.params.epsilon,self.params.w) < 0:
            raise ValueError('invalid HMM parameters')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', InconsistentVersionWarning)
            self.model = joblib.load(self.model_path)
        self.version_warnings = [str(w.message) for w in caught if issubclass(w.category, InconsistentVersionWarning)]
        for warning in caught:
            if not issubclass(warning.category, InconsistentVersionWarning):
                warnings.warn(str(warning.message), warning.category, stacklevel=2)
        if self.version_warnings and not allow_version_mismatch:
            raise RuntimeError('Model/runtime sklearn versions differ. Use the training runtime; diagnostic override must be explicit.')
        classes = list(self.model.classes_)
        if set(classes) != {0, 1} or len(classes) != 2:
            raise ValueError('binary model classes [0,1] required')
        self.positive_index = classes.index(1)
        if getattr(self.model, 'n_features_in_', len(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
            raise ValueError('model feature count mismatch')

    def predict_trip(self, trip_dir, return_details=False):
        start = time.perf_counter()
        trip = load_prediction_trip(trip_dir)
        cands = prediction_candidates(trip)
        if cands.empty:
            raise ValueError('No candidate arcs for this trip; cannot produce a model route')
        features = build_features(trip, cands)
        p = np.asarray(self.model.predict_proba(features[self.feature_columns]))[:, self.positive_index]
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('model emitted invalid candidate probabilities')
        scored = cands.copy()
        scored['p_match'] = p
        graph = RoadGraph(trip.nodes, trip.arcs)
        # Binary per-candidate probabilities: do NOT normalize within observation.
        predicted = decode(cands, np.log(np.clip(p, 1e-6, 1.0)), graph,
                           params=self.params, obs_times=trip.track[:, 2], n_obs=len(trip.track))
        if (predicted < 0).any() or (predicted >= len(trip.arcs)).any():
            raise ValueError('decoder could not fill every observation with a valid arc')
        runs = predicted[np.r_[True, predicted[1:] != predicted[:-1]]]
        coords = []
        segments = []
        for aid in runs:
            u, v = trip.arcs[aid]
            segment = [trip.nodes[u, ::-1].tolist(), trip.nodes[v, ::-1].tolist()]
            segments.append(segment)
            coords.extend(segment)
        result = {'trip_id':trip.trip_id, 'n_obs':len(trip.track),
                  'gps':trip.track[:, [1, 0]].tolist(),
                  'predicted_arcs':predicted.tolist(), 'predicted_path':coords,
                  'predicted_segments':segments,
                  'path_semantics':'endpoint pairs in run order, not a connected polyline; render predicted_segments separately',
                  'n_empty_candidate_observations':int(len(trip.track)-cands.obs_idx.nunique()),
                  'metrics':{'top1':None,'macro_f1':None,'route_continuity':None,'sequence_f1':None},
                  'latency_ms':round((time.perf_counter()-start)*1000, 3),
                  'mock':False, 'prediction_mode':'model_hmm',
                  'warnings':self.version_warnings.copy() + ([self.provenance['limitation']] if self.provenance else []),
                  'model_provenance':self.provenance, 'hmm_params':vars(self.params)}
        # Each pair is an actual arc. Consumers should not invent connector roads
        # between disconnected pairs. Folium renderer draws each arc separately.
        if return_details:
            return result, trip, scored, graph
        return result


def predict_trip(trip_dir, predictor=None, **model_options):
    """Public entry point; pass a Predictor to avoid reloading the model."""
    return (predictor or Predictor(**model_options)).predict_trip(trip_dir)
