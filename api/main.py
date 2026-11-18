"""API-01: development API skeleton, NOT a trained-model predictor.

Windows cmd (repository root):
set EVAL_DATA_DIR=C:\\path\\to\\trip_folders
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Only .track files are read. Reference routes never enter this stub or response.
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


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    trip_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


def create_app(data_dir=None, log_path=None):
    """Build an isolated app; env is read at startup, not per request.

    EVAL_DATA_DIR must contain immediate trip folders, each with <id>.track.
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
            yield
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()

    app = FastAPI(title="PathSnap API-01 GPS stub", lifespan=lifespan,
                  description="UI integration only. No road prediction or model scores.")
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

    def track_path(root, trip_id):
        folder = root / trip_id
        # Never follow a trip or track symlink outside the configured root.
        if folder.is_symlink() or not folder.is_dir():
            return None
        path = folder / (trip_id + ".track")
        if path.is_symlink() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(root):
            return None
        return path

    @app.get("/health")
    def health():
        return {"status": "ok", "prediction_mode": "gps_stub", "mock": True}

    @app.get("/trips")
    def trips():
        root = root_dir()
        return {"trips": sorted(p.name for p in root.iterdir()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", p.name)
            and track_path(root, p.name) is not None)}

    @app.post("/predict")
    def predict(body: PredictionRequest, request: Request):
        start = time.perf_counter()
        request.state.trip_id = body.trip_id
        path = track_path(root_dir(), body.trip_id)
        if path is None:
            raise HTTPException(404, "Trip track not found")
        try:
            track = np.loadtxt(path, ndmin=2)
        except (ValueError, OSError):
            raise HTTPException(422, "Trip track is unreadable or malformed")
        if track.size == 0 or track.shape[1] != 3 or not np.isfinite(track).all():
            raise HTTPException(422, "Track must contain finite longitude, latitude, timestamp rows")
        if (np.abs(track[:, 0]) > 180).any() or (np.abs(track[:, 1]) > 90).any():
            raise HTTPException(422, "Track coordinates are out of range")
        gps = track[:, [1, 0]].tolist()  # API/UI order is latitude, longitude.
        return {"trip_id": body.trip_id, "n_obs": len(track), "gps": gps,
                "predicted_arcs": [], "predicted_path": gps,
                "metrics": {"top1": None, "macro_f1": None,
                            "route_continuity": None, "sequence_f1": None},
                "latency_ms": round((time.perf_counter()-start)*1000, 3),
                "mock": True, "prediction_mode": "gps_stub",
                "warning": "GPS passthrough only: no model or HMM prediction; metrics unavailable"}

    return app


app = create_app()
