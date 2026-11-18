# Initial model configurations: M1-M3

TRAIN-02 configuration record. These are the starting settings for the
first-pass comparison, not tuned results. Seed is 42 unless the runner passes
another seed. The selected model may receive different settings during BO-01.

| ID | Model | Family | Initial configuration | Scaling | Row cap per training fold |
|---|---|---|---|---|---:|
| M1 | Logistic Regression | Linear | `C=1.0`, `class_weight="balanced"`, `max_iter=1000`, `random_state=42` | StandardScaler | 300,000 |
| M2 | Gaussian Naive Bayes | Probabilistic | scikit-learn defaults | None | 300,000 |
| M3 | Random Forest | Ensemble (bagging) | `n_estimators=100`, `max_depth=12`, `min_samples_leaf=20`, `random_state=42`, `n_jobs=-1` | None | 100,000 |

## Shared preprocessing

The training harness wraps each estimator in a scikit-learn pipeline with
median imputation. For M1, standard scaling runs after imputation. Because
these steps are inside the pipeline, their statistics are learned on the
training fold only. M2 and M3 use imputation without scaling.

## Probability output

All three classifiers support `predict_proba`. The positive-class probability
for label `1` is used as `p_match` by the route-decoding stage. Keep the class
order explicit in the runner by selecting the column whose class is `1`.

## Notes

- The four-minute limit applies to each pass-1 model run; the row caps above
  come from the team sprint plan.
- `max_iter=1000` is set for Logistic Regression to give the solver enough
  iterations on the feature table. Check convergence warnings in the run log.
- No performance values are included here. Record only measurements from
  actual training runs in `results/`.
