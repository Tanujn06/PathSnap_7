# TRAIN-02: M1-M3 model definitions

Copy the contents of this ZIP into the root of the PathSnap repository.

Files:
- `training/models_m1_m3.py` — model factories and metadata used by TRAIN-03.
- `docs/MODEL_CONFIGS_M1_M3.md` — initial hyperparameters and preprocessing notes.
- `tests/test_models_m1_m3.py` — model contract and small CV smoke tests.

Run from the repository root with the TRAIN-01 virtual environment active:

```bash
python -m pytest -q tests/test_models_m1_m3.py
```

The tests use synthetic data only. They confirm that all three estimators
support probability predictions and can pass through the common five-fold
training harness. They are not model performance results.
