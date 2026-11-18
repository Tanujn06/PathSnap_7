# Initial model configurations: M4 to M6

Owner: Sinchana (TRAIN-04). Source of truth: `training/models_m4_m6.py` (`build_models()`;
keys `knn`, `hist_gradient_boosting`, `mlp`, same interface as M1-M3).
The shared harness adds median imputation to every model and StandardScaler when `scale=True`.
Seed 42. CV: GroupKFold(5) by `trip_id` on the 80 train trips.
Task: binary classification of (GPS point, candidate arc) rows, output `p_match`.

| Model | Family | Preprocessing | Initial hyperparameters | Pass-1 row cap / fold | Loss / objective |
|---|---|---|---|---|---|
| M4 KNN | Instance-based | Median impute + StandardScaler | n_neighbors=15, weights=distance, Euclidean (minkowski p=2) | 100,000 | None (lazy learner); probability = distance-weighted vote of the 15 neighbours |
| M5 HistGradientBoosting | Boosting | Median impute only (trees are scale-invariant) | learning_rate=0.1, max_iter=200, max_leaf_nodes=31, early_stopping=True (10% validation, patience 10) | 300,000 | Binary log loss (binary cross-entropy) |
| M6 MLP | Neural network | Median impute + StandardScaler | hidden=(64, 32), ReLU, Adam lr=1e-3, alpha=1e-4, batch 256, max_iter=100, early_stopping=True (10% validation, patience 10) | 100,000 | Binary log loss (cross-entropy) + L2 penalty alpha |

Notes
- Only M6 produces an epoch-wise training history (`pipe.named_steps["model"].loss_curve_`, `.validation_scores_`).
  No training curves are drawn for M4 or M5.
- Scalers sit inside the Pipeline, so they are fitted on the training folds only.
