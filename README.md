# PathSnap

Distributed map-matching and route reconstruction for the UE24CS352A Machine
Learning Hackathon (18 Nov 2026).

Given raw GPS traces and a per-trip road network, PathSnap scores candidate
road segments per GPS point with a scikit-learn classifier and decodes the
most consistent route with an HMM/Viterbi stage, served through a FastAPI
backend (Laptop 1) and a plain HTML/Leaflet UI (Laptop 2).

## Architecture

```
Laptop 2 (UI)  --HTTP-->  Laptop 1 (FastAPI)  -->  candidates -> model -> Viterbi -> route JSON
```

- **Laptop 1** hosts the trained classifier, the HMM/Viterbi decoder, the road
  networks and the API.
- **Laptop 2** shows the ten evaluation trips in a dropdown, sends the
  selected trip to the API and renders the returned route on a map.

## Requirements

Python 3.11+, CPU only (no GPU libraries, no LightGBM/XGBoost).

```bash
pip install -r requirements.txt
```

Stack: numpy, pandas, scipy, pyarrow, scikit-learn (pinned to 1.7.2), optuna, fastapi, uvicorn, folium, matplotlib, joblib, pytest and httpx2.

## Dataset setup

Unzip the dataset into `Student folder/` so that
`Student folder/train/<trip_id>/` and `Student folder/student_test/<trip_id>/`
each contain the four trip files (`*.track`, `*.nodes`, `*.arcs`, `*.route`).
Raw data and `data_cache/` outputs are never committed; caches are regenerated
by the commands below.

## Regenerating the pipeline caches

From the repository root, with `PYTHONPATH=src` (Windows: `set PYTHONPATH=src`):

```bash
# Candidates per GPS point (DATA-02), one split at a time
python -m pathsnap.candidates --data-root "Student folder" --split train
python -m pathsnap.candidates --data-root "Student folder" --split student_test

# Reference alignment labels, development only (DATA-04), both splits
python -m pathsnap.labels --data-root "Student folder"

# Feature tables (FEAT-01): data_cache/features_{train,student_test}.parquet
python -m pathsnap.features --data-root "Student folder"
```

Each trip takes seconds; the full train split is minutes.

## Training, comparison and export

```bash
# Pass-1 runs (GroupKFold-5 by trip, seed 42, plan row caps)
python training/run_m1_m3.py     # appends results/runs_m1_m3.csv
python training/run_m4_m6.py     # appends results/runs_m4_m6.csv

# Bayesian optimisation of the selected model (Optuna TPE; search log retained)
python training/bayes_opt.py     # writes results/ optimisation records

# Final refit on all rows and export
python training/export_final.py  # writes models/final.joblib + models/config.json
```

The exported final model is the HistGradientBoosting classifier, chosen for
the best candidate top-1 accuracy on the held-out student split (0.8850 of
M1-M6), the score the HMM decoder consumes. The full comparison is in
results/model_comparison.csv; metrics are defined exactly in
docs/METRICS.md (EVAL-01).

## Running the API (Laptop 1)

```bash
export EVAL_DATA_DIR=/path/to/evaluator_test   # folder with the 10 trip folders
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints (contract in docs/CONTRACT.md): `GET /health`, `GET /trips`,
`POST /predict`. Every request is logged to the console and `api.log`
(time, trip_id, latency).

## Running the UI (Laptop 2)

```bash
scripts/run_ui.sh          # serves ui/ on http://localhost:5500
scripts/run_ui.sh 6000     # or pick another port
```

By default the UI calls the API at `http://127.0.0.1:8000`. To point it at
Laptop 1 without editing code, open
`http://localhost:5500/?api=http://<laptop-1-ip>:8000`; `?mock=1` forces the
offline mock files instead. Pick one of the ten trips and the decoded route is
drawn on the Leaflet map. Stub or mock responses are always labelled on
screen.

## Running the tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

62 tests cover loaders, candidates, graph, labels, features, models,
metrics, the decoder, and API endpoints.

## Layout

- `src/pathsnap/` - data pipeline, HMM decoder, evaluation, API pipeline
- `training/` - harness, model scripts M1-M6, Bayesian optimisation, export
- `api/` - FastAPI backend
- `ui/` - plain HTML + JavaScript UI with vendored Leaflet
- `viz/` - EDA, Folium maps, comparison plots
- `configs/` - frozen feature column order and HMM parameters
- `models/` - exported final model and configuration
- `results/` - run CSVs, optimisation records, plots, maps
- `docs/` - data/API contract, feature contract, metric definitions, assumptions

## Branching

`main` is always demoable and is updated only by the release pull request
from `dev`. All work happens on `feature/`, `chore/`, `docs/` and `fix/`
branches that merge into `dev` through pull requests. Never commit `data/`,
passwords or keys.
