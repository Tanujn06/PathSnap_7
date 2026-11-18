"""API-02: real reference-free model + HMM inference.

Windows cmd (repository root):
set PYTHONPATH=src
set EVAL_DATA_DIR=C:\\path\\to\\trip_folders
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Reference routes are optional, server-side evaluation only after prediction.
"""
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import f1_score
from pathsnap.pipeline import Predictor
from pathsnap.data_io import Trip
from pathsnap.labels import align_route
from pathsnap.evaluate import route_continuity, sequence_level_f1


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    trip_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


def create_app(data_dir=None, log_path=None, predictor_factory=Predictor):
    """Build an isolated app; env is read at startup, not per request.

    EVAL_DATA_DIR must contain immediate trip folders, each with <id>.track/.nodes/.arcs.
    API_LOG_PATH defaults to api.log. Failure to open it fails startup rather
    than claiming to log requests that were never saved.
    """
    logger = logging.getLogger("pathsnap.api." + uuid4().hex)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    @asynccontextmanager
    async def lifespan(app):
        configured = data_dir if data_dir is not None else os.environ.get("EVAL_DATA_DIR")
        app.state.data_dir = Path(configured).expanduser().resolve() if configured else None
        target = Path(log_path if log_path is not None else os.environ.get("API_LOG_PATH", "api.log"))
        target.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(target, encoding="utf-8")]
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        try:
            app.state.predictor = predictor_factory()
            yield
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()

    app = FastAPI(title="PathSnap API-02 model + HMM", lifespan=lifespan,
                  description="Reference-free inference with optional post-prediction server evaluation.")
    # Development LAN testing only. No cookies/auth credentials are accepted.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            logger.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                "method": request.method, "path": request.url.path,
                "trip_id": getattr(request.state, "trip_id", None),
                "status": status, "latency_ms": round((time.perf_counter()-started)*1000, 3)},
                ensure_ascii=True))

    def root_dir():
        root = app.state.data_dir
        if root is None or not root.is_dir():
            raise HTTPException(503, "Set EVAL_DATA_DIR to an existing folder containing trip folders")
        return root

    def trip_folder(root, trip_id):
        folder = root / trip_id
        if folder.is_symlink() or not folder.is_dir():
            return None
        for ext in ('track', 'nodes', 'arcs'):
            path = folder / (trip_id + '.' + ext)
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                return None
        return folder

    @app.get("/health")
    def health():
        return {"status": "ok", "prediction_mode": "model_hmm", "mock": False,
                "model_provenance": app.state.predictor.provenance}

    @app.get("/trips")
    def trips():
        root = root_dir()
        return {"trips": sorted(p.name for p in root.iterdir()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", p.name)
            and trip_folder(root, p.name) is not None)}

    @app.post("/predict")
    def predict(body: PredictionRequest, request: Request):
        start = time.perf_counter()
        request.state.trip_id = body.trip_id
        folder = trip_folder(root_dir(), body.trip_id)
        if folder is None:
            raise HTTPException(404, "Trip prediction inputs not found")
        try:
            result, trip, scored, graph = app.state.predictor.predict_trip(folder, return_details=True)
        except (ValueError, OSError) as exc:
            logger.warning(json.dumps({"event":"prediction_failed", "trip_id":body.trip_id,
                                       "error_type":type(exc).__name__}))
            raise HTTPException(422, "Trip inputs cannot produce a valid prediction") from exc
        # No reference access occurs until prediction has returned in full.
        result['metrics'] = dict(top1=None, macro_f1=None,
                                route_continuity=route_continuity(result['predicted_arcs'], graph),
                                sequence_f1=None)
        result['evaluation_status'] = 'reference_unavailable'
        reference = folder / (body.trip_id + '.route')
        if reference.is_file() and not reference.is_symlink():
            try:
                raw = np.loadtxt(reference, ndmin=1)
                if not np.isfinite(raw).all() or (raw != np.floor(raw)).any():
                    raise ValueError('invalid reference arc indices')
                route = raw.astype(np.int64)
                labels = align_route(Trip(trip.trip_id, trip.track, trip.nodes, trip.arcs, route))
                truth = labels.ref_arc.to_numpy()
                predicted = np.asarray(result['predicted_arcs'])
                result['metrics'].update(top1=float(np.mean(predicted == truth)),
                    macro_f1=float(f1_score(truth, predicted, average='macro', zero_division=0)),
                    sequence_f1=sequence_level_f1(predicted, route))
                result['evaluation_status'] = 'reference_evaluated_after_prediction'
            except (ValueError, OSError, IndexError):
                result['evaluation_status'] = 'invalid_reference_ignored'
        result['metric_semantics'] = {
            'top1':'HMM arc accuracy, all observations, derived alignment labels',
            'macro_f1':'HMM per-observation multiclass macro F1, union of truth/predicted arc classes',
            'route_continuity':'whole-trip directed direct transitions after repeated arcs collapsed',
            'sequence_f1':'arc-set overlap F1, ignores traversal order'}
        # Keep the frozen flat predicted_path contract. Clients should prefer
        # predicted_segments to render independent arcs without invented bridges.
        result['warning'] = '; '.join(result['warnings'])
        if result['evaluation_status'] != 'reference_evaluated_after_prediction':
            result['warning'] += '; Reference metrics unavailable; prediction is still real model + HMM'
        result['latency_ms'] = round((time.perf_counter()-start)*1000, 3)
        return result

    return app


app = create_app()
