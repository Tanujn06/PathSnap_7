"""API-02: reference-free model + HMM prediction API.

Run from the repository root:
export EVAL_DATA_DIR=/path/to/trip_folders
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000

Inference requires .track, .nodes, and .arcs files per trip.
The reference .route file is not read during prediction.
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
from pathsnap.pipeline import Predictor


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    trip_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


def create_app(data_dir=None, log_path=None, predictor=None):
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
        app.state.predictor = predictor if predictor is not None else Predictor()
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

    app = FastAPI(title="PathSnap API-02 Model Predictor", lifespan=lifespan,
                  description="Reference-free road prediction using the trained model and HMM.")
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
        ready = getattr(app.state, "predictor", None) is not None
        return {
            "status": "ok" if ready else "unavailable",
            "prediction_mode": "model_hmm" if ready else "unavailable",
            "mock": False,
        }

    @app.get("/trips")
    def trips():
        root = root_dir()
        return {"trips": sorted(p.name for p in root.iterdir()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", p.name)
            and track_path(root, p.name) is not None)}

    @app.post("/predict")
    def predict(body: PredictionRequest, request: Request):
        request.state.trip_id = body.trip_id
        path = track_path(root_dir(), body.trip_id)
        if path is None:
            raise HTTPException(404, "Trip track not found")

        try:
            result = app.state.predictor.predict_trip(path.parent)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, f"Prediction input or processing error: {exc}")
        except RuntimeError:
            logger.exception("Prediction runtime unavailable")
            raise HTTPException(503, "Prediction model/runtime unavailable")

        return result

    return app


app = create_app()
