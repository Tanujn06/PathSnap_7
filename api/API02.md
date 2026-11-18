# API-02: real prediction integration

Uses PIPE-01's selected sklearn1.7.2 resaved HGB + tuned HMM, loaded once at
startup. Original model and training provenance are unchanged. Startup fails
if the model hash/runtime guard fails. There is no GPS-stub fallback.

## Windows cmd

Run from the repository root:

```cmd
python -m pip install -r requirements.txt
python -m pip install scikit-learn==1.7.2 httpx
set PYTHONPATH=src
set EVAL_DATA_DIR=C:\path\to\Student folder\student_test
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second cmd window from the same repository:

```cmd
python -m http.server 5500 --directory ui
```

Open `http://127.0.0.1:5500/?api=http://127.0.0.1:8000` on the same machine, or
replace the API address with Laptop1's actual LAN address on Laptop2. This is
a development LAN server with open CORS, not an authenticated production API.

Tests:

```cmd
set PYTHONPATH=src
python -m pytest -q
```

## Boundary and response

`POST /predict` calls Predictor using `.track/.nodes/.arcs` only. It reads an
optional server-side `.route` only after inference returns and never includes
the reference sequence or labels in the response. Missing/invalid references
do not invalidate a real prediction. Missing trip inputs return404; malformed
prediction inputs/no candidate arcs return422. Symlinked trips/required files
are not exposed. `/health` reports `model_hmm`, `mock=false`, model provenance.
Requests, trip IDs, statuses and latency continue to be logged.

Flat `gps` and `predicted_path` retain the frozen lat/lon contract. The additive
`predicted_segments` contains independent actual-arc endpoint pairs. The UI
prefers these separate lines, preventing invented connectors across route
gaps, and retains flat fallback for older stub/mock responses. A single arc
also renders. API/UI files do not alter model/decoder artifacts.

## Metric definitions

Metrics are evaluated AFTER HMM prediction, independently for each requested
trip. The UI shows that trip only. No split aggregate is substituted:

- `top1`: HMM observation arc accuracy, all observations, DATA-04 derived
  monotone-alignment reference labels, not physical ground truth.
- `macro_f1`: HMM observation multiclass macroF1 over the union of predicted
  and reference arc IDs, absent per-class divisions0. This is NOT the binary
  candidate-classifier macroF1 used in training reports.
- `route_continuity`: direct directed transitions after adjacent repeated arcs
  collapse, across the whole trip including gaps. No reference needed.
- `sequence_f1`: arc-set overlap, ignoring traversal order/repeat counts.

Without a usable reference, top1/macroF1/sequenceF1 are null; continuity remains
computed. `evaluation_status` says why. Model warnings and source provenance
remain in the JSON; `warning` is also shown by the existing UI. Test labels do
not enter candidate generation, feature building, model scoring or decoding.

## Acceptance

62tests passed, including5API tests (success, missing/malformed inputs,
reference-read ordering, startup failure, symlinks). All10student_test trips
returned200with actual model/HMM arcs and postprediction metrics. Actual browser
UI on trip00000052displayed262observations, top1/macroF1=1.000,
continuity0.877,setF1=0.927, real redroutearcs, visible runtime limitation,
noMOCKbadge and noJavaScript errors. Evidence records actual measured latency,
not a cross-machine performance guarantee. The accepted resave is tested under
sklearn1.7.2; original1.9.1execution equivalence is still unproved.
