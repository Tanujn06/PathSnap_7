# UI-03 screenshots

Captured 18 Nov 2026 from the UI on branch dev (after API-02, PR #30), in
headless Chrome at 1400x850.

Setup: real API (`api.main:app`, prediction_mode `model_hmm`, mock false) with
scikit-learn 1.7.2 and the resaved final model, `EVAL_DATA_DIR` = the 10
student_test trips; UI served by `scripts/run_ui.sh` on port 5500 and opened as
`http://localhost:5500/?api=http://127.0.0.1:8000&trip=<id>`.

Both the API and the UI ran on Laptop 2 for these captures. The two-laptop
network setup (UI on Laptop 2, API on Laptop 1 over the LAN) is checked
separately at the rehearsal.

| File | Shows |
|---|---|
| `trip_<id>.png` (10 files) | Raw GPS, predicted route segments, post-prediction metrics and API latency for each student_test trip |
| `mock_fallback.png` | `?mock=1` offline fallback: MOCK DATA badge and warning, no metrics |
| `api_unreachable.png` | API stopped: error message, Predict disabled |

All 11 `/predict` calls returned HTTP 200 (api.log). Metrics are computed per
trip after prediction against the derived reference labels (see api/API02.md);
the orange note is the API's scikit-learn version limitation (models/runtime.json).
