# EVAL-01 metrics and interfaces

This library evaluates development results. It does not predict routes, read evaluator labels, train models or change the feature/label tables.

Arc IDs throughout are the actual zero-based raw `.arcs` row IDs used by the road graph, candidates and DATA-04 labels, not names, ordinal positions of collapsed runs or reference-route positions.

## Interfaces

- `macro_f1_candidates(y_true, y_pred)`: one binary class per candidate row. Mean F1 over fixed classes `[0,1]`, with undefined/absent class F1 set to zero. Classification probabilities must be thresholded at `p_match >= 0.5` before this function.
- `log_loss_candidates(y_true, p_match)`: binary negative natural-log likelihood averaged over all candidate rows. Validate values in `[0,1]`, then clip to `[1e-7, 1-1e-7]` for numerical stability.
- `validate_probabilities(features, probabilities)`: require unique, nonnull `(trip_id, obs_idx, arc_id)` keys and EXACTLY the same key set in both frames. Scores join by keys, not row order. `trip_id` must remain a string. Missing, duplicate, extra or invalid-probability rows fail loudly. No silently filling missing model scores.
- `top1_predictions(scored)`: largest p_match for each `(trip_id, obs_idx)`; ties choose the smallest arc ID. Returns `trip_id, obs_idx, pred_arc`.
- `top1_obs_accuracy(scored, labels)`: correct chosen reference arc count divided by ALL reference-labelled GPS observations. Missing-candidate observations count as incorrect. This does not measure agreement with independently verified physical road truth; labels were derived by DATA-04.
- `candidate_miss_rate(candidates, labels)`: number of all labelled observations whose aligned reference arc is absent from candidate rows divided by all labelled observations. Empty-candidate observations are misses. Candidate recall = 1 minus this miss rate. Recall, NOT miss rate, is the ceiling on top-1 for a classifier restricted to these candidates. A decoder that inserts noncandidate arcs is not covered by that ceiling.
- `continuity_counts(predicted_arcs, graph)`: per-trip integer arc array plus Simon's `RoadGraph` (or object with its `arcs` array). Collapse adjacent repeated arc IDs into runs, then count direct directed head-to-tail transitions. Return connected count, transition count and ratio. `route_continuity` returns just that ratio. It uses direct node equality rather than cached multi-hop results, so a two-hop transition cannot accidentally pass as one hop. Reversed direction does not count unless head-to-tail adjacency actually exists. Single-arc/stationary nonempty paths score 1 (zero invalid transitions); empty paths return None, not a fabricated perfect route.
- `sequence_level_f1(predicted_arcs, reference_route)`: `2 * |predicted_set intersection reference_set| / (|predicted_set| + |reference_set|)`. Both empty = 1, exactly one empty = 0. This ignores order and repeated-arc multiplicity; it is set-overlap F1, not edit distance or sequence alignment.
- `summary(features, probabilities, labels)`: validates G4 key alignment and returns candidate-row count, all-observation count, top1, macro_f1, log_loss, candidate miss rate and recall. Route metrics remain explicit per-trip calls because each trip has its own road graph/reference route. Never concatenate different trips into one route. Never bridge an empty-observation gap by dropping its index: score segments separately or use the decoder's explicitly filled sequence, and state the method.

## Plan wording resolved explicitly

The plan says both "consecutive distinct" arcs and "same arc" when describing continuity. We use consecutive distinct runs, so repeated samples on a stationary arc do not inflate the denominator. This choice is explicit and must be identical for every model and before/after-HMM comparison. Do not compare this number to an observation-transition denominator without relabelling the metric.

Macro F1 is candidate-level binary classification. Route-decoded predictions do not have candidate-row class probabilities, so macro F1/log loss are not automatically defined for a decoded route. Use N/A with explanation or define and document an additional derived classification scheme separately. Do not invent scores from the route sequence.

No automatic residual filtering is applied. Report any >50m subset as a separate diagnostic with its denominator, alongside all-observation metrics; do not substitute it for the main result. Fit learned preprocessing on training folds only, outside this metrics library.

## Verification actually performed

Eight tests on hand-made examples pass: hand-calculated F1/log loss/top1, missing candidate, tied scores, unordered probability input, missing/duplicate keys, invalid probability/arc inputs, direct versus two-hop graph edges, repeated arcs, empty cases and absent binary classes. These are synthetic TESTS, not model experiments.

Real full-dataset `candidate_miss_rate` rerun on the frozen FEAT-01 tables gives train `16147/208417 = 0.07747448624632347`; student-test `46/19160 = 0.0024008350730688934`. Train includes 198 empty-candidate observations. These agree with feature diagnostics, measure candidate coverage only and are not new trained-model results.

No real trained-model probabilities were provided during this implementation. Therefore no model top1, macro F1, log loss, HMM improvement or model-comparison conclusion is claimed here. G4 model producers can call `summary` once their actual probability parquets exist.

Run from repository root:

```
set PYTHONPATH=src
python -m pytest tests/test_evaluate.py -q
```
