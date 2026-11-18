# TRAIN-01 - training harness

This adds the common helpers used by the M1-M6 runners.

- `make_mock()` creates a small fake feature table so the harness can be tested before Tanuj finishes FEAT-01.
- `load_features()` checks the agreed ID/label columns.
- `split_xy()` keeps identifiers out of the model features.
- `make_pipeline()` puts imputation/scaling inside each sklearn fold.
- `fit_cv()` uses `GroupKFold(5)` grouped by `trip_id` and records fold metrics/timing.
- `append_run()` and `save_probabilities()` are small output helpers for the next training task.

Run from the repository root:

```bash
python -m training.mock_data
pytest -q tests/test_training_harness.py
```

Mock rows are only for a smoke test. They must not be included in reported hackathon results.
