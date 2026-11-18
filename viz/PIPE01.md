# PIPE-01: reference-free prediction and Folium maps

Scope: inference and development maps only. API-02 is separate. No model fitting,
HMM tuning, historical result overwrite, root README or report-section edits.

## Selected model and provenance

The user chose the final-model sklearn1.7.2 resave on2026-11-18. Original
`models/final.joblib` remains unchanged (sklearn1.9.1 serialization). The default
predictor uses separately named `models/final_sklearn172.joblib`, selected by
`models/runtime.json`. The runtime config records both hashes and the original
training provenance. Both files' hashes are checked before default loading.
Default loading requires the tested sklearn1.7.2 runtime. A metadata-only resave
is never treated as proof of original-runtime equivalence.

Independent verification on all155598canonicalstudentcandidate rows found the
original-loaded and resaved models' `predict_proba` outputs bit-identical in
sklearn1.7.2 (maximum absolute difference0). Reload emitted no version warnings.
No fitting or coefficient changes were performed. This establishes same-runtime
resave identity, NOT equivalence with execution in sklearn1.9.1. Every default
prediction and map records this limitation. Only load trusted joblib files,
since pickle-based formats can execute code.

The original model config still owns the frozen13feature order/training details.
`configs/hmm_params.json` owns tuned lambda4, epsilon100, gap10s parameters.
Candidate probabilities are binary per-candidate probabilities, not normalized
within observation.

## Windows cmd usage

Install the project requirements using the selected tested sklearn version:

```cmd
python -m pip install -r requirements.txt
python -m pip install scikit-learn==1.7.2
set PYTHONPATH=src
python -c "from pathsnap.pipeline import predict_trip; r=predict_trip(r'Student folder\student_test\00000052'); print(r['predicted_arcs']); print(r['model_provenance'])"
python viz\folium_map.py --data-root "Student folder\student_test"
python -m pytest -q
```

`Predictor` loads once for repeated trips. The convenience `predict_trip` loads
project-relative defaults independent of the caller's working directory.
Prediction reads `.track`, `.nodes`, `.arcs` only, never `.route`, including when
an invalid reference file is present. Missing/malformed inputs and trips with
no candidate arcs fail explicitly rather than returning a fake route.

Outputs contain one arc per observation and coordinates in latitude/longitude
order. Draw `predicted_segments` separately: each is an actual arc endpoint pair.
The flattened `predicted_path` is for inspection, not a connected polyline. No
connector roads are invented. Empty-candidate observations follow the existing
HMM nearest-decoded-arc filling behavior; their count is returned. Inference
metrics are null because reference labels are unavailable.

Explicit custom model paths retain version-warning refusal. A diagnostic
`allow_version_mismatch=True` exists for controlled original-model comparison,
but does not override the selected resave's explicit runtime/hash guard.

## Maps and evidence

Map generation completes inference before loading reference routes for separate
development evaluation. HTML maps need network access for Leaflet assets and
OpenStreetMap tiles. Layers: blue GPS dots, grey bbox-clipped roads, green
reference route, red predicted arcs, controls and legend. Arc gaps stay gaps.

Three post-hoc illustrative cases are chosen deterministically: highest HMM
accuracy; largest HMM improvement among remaining trips; then lowest HMM accuracy
among the remainder. These choices are not used for tuning. The failure case is
partial failure, not total decoder failure.

- Good00000052: classifier0.969466, HMM1.000000.
- HMM-helped00000091: classifier0.863844, HMM0.948513.
- Partial failure00000077: classifier0.747379, HMM0.828433.
- All10trips: classifier0.879280, HMM0.929854 across19160observations.
- Candidate macroF1:0.926253 across155598candidate rows.

`selection.json` records case selection, params and model provenance;
`pipeline_student_test.csv` records every trip, not just chosen examples.
`pipeline_probabilities.parquet` and `pipeline_predictions.parquet` contain full
keyed outputs. `runtime_audit.json`, `resave_verification.json`, and
`acceptance.json` record independent checks and their precise scope.

Labels are DATA-04 derived monotone alignment, not physical truth. Top1 uses all
19160observations. Whole-trip continuity counts direct directed transitions
with adjacent repeated arcs collapsed, including transitions across GPS gaps.
Set F1 ignores order, so accuracy1.0 need not mean continuity1.0.

Historical `hmm_student_test_predictions.parquet` is NOT overwritten. Simon's
four-combination rerun reproduced that historical00000007output99.93%using
untuned pass2 HGB/default HMM; tuned final-model/tuned HMM exactly matches PIPE01.
This is rerun-based provenance, not an original generation log. New outputs are
separate and identify the selected model/config.

## Tests

The57-test suite includes11PIPE01tests: success without reference, refusal to
open invalid reference, upstream candidate parity on fixture, missing inputs,
no candidates, malformed inputs, feature-order check version handling, selected-resave hash/runtime/provenance guards.
Test classifiers are tiny dummy fixtures, not experiment evidence. Actual maps
use the accepted final-model resave. Real262observation00000052acceptance was
also run with no route and an invalid route, with identical results.
