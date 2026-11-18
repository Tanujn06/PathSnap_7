# FEAT-01 feature contract and run evidence

Run from repository root:

```
set PYTHONPATH=src
python -m pathsnap.features --data-root "data/Student folder" --cache data_cache
python -m pytest tests/test_features.py -q
```

If `train` and `student_test` are directly inside `data`, use `--data-root data`.
Required inputs: the unchanged DATA-02 candidates and DATA-04 labels parquet files in `data_cache`, plus the raw trips. Output: `features_train.parquet` and `features_student_test.parquet` in the same cache directory. Cache files remain gitignored.

## Frozen schema

`trip_id, obs_idx, arc_id, label`, followed by the 13 features in `configs/feature_columns.json`. At inference, omit the optional labels argument and no label column is generated. Features do not inspect `trip.route` or derive reference labels.

Features: distance in metres, along-arc fraction, zero-based candidate rank, distance minus minimum, arc length in metres, GPS heading in degrees clockwise from north, heading-to-arc cosine, speed in metres/second, time since previous point, number of supplied candidates within 20m, in-degree of arc start node, out-degree of arc end node, reverse arc existence.

Motion: edge-padded 3-point coordinate moving average, followed by centred neighbour differences (available neighbours at endpoints). No learned preprocessing is fit here. Stationary or zero-length-arc directional evidence has cosine 0. Speed is 0 for nonpositive elapsed time; actual dt is retained. Very large gaps are not silently clipped. Direction and speed can be unreliable around long gaps and stops. Counting within 20m is limited to the supplied capped candidate set, not an exhaustive neighbourhood census.

## Full real run

| Split | Candidate/feature rows | Observations in recall denominator | Reference found | Missed | Empty-candidate observations |
|---|---:|---:|---:|---:|---:|
| train | 1664104 | 208417 | 192270 | 16147 | 198 |
| student_test | 155598 | 19160 | 19114 | 46 | 0 |

Train recall 92.252551%, miss 7.747449%. Student-test recall 99.759916%, miss 0.240084%. Denominator is every labelled observation, including empty-candidate cases. Recall measures whether the derived reference arc is available for classification; it is not model accuracy. The training recall is below the planned >95% target. No candidate-generation change, row exclusion or reference-label replacement was made. Consult `docs/ASSUMPTIONS.md` for reference coverage and alignment caveats; candidate misses and labels should not be presented as verified road truth.

All 1819702 supplied candidate rows are preserved, features are finite, and column order is fixed. Six feature tests and five labels tests pass. No model-training score is claimed. Trip IDs preserve leading zeros as strings; keys remain separate from input features so the trainer can exclude identifiers and label leakage. The current training harness accepts this schema.

## Transferred cache chunks

For transfer only, training rows can be divided into part parquet files. Reconstruct by concatenating parts in filename order with pandas and writing `data_cache/features_train.parquet`. The part order preserves the original feature-table row order. Student-test table is supplied intact. `combine_feature_parts.py` performs this reconstruction and validates frozen column order. This does not change training/test partitions.
