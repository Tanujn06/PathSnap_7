"""harness.py - training harness (TRAIN-01).

Owner: Siddharth. Branch: feature/training-harness.
GroupKFold(5) by trip_id over the features table; sklearn Pipelines so
scalers fit inside each fold only; per-run rows appended to results/runs_*.csv.
"""
