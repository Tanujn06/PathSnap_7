## 5. HMM decoding: where it helps and where it does not

### 5.1 Overall effect

Decoding the classifier's per-candidate probabilities with the HMM improves candidate top-1 accuracy on training out-of-fold predictions from 0.7773 (argmax baseline) to 0.8184 with default transition parameters, and to 0.8267 after tuning (lambda_per_hop 4.0, epsilon 100, gap_threshold_s 10), a gain of +4.94 percentage points over baseline. Tuning used only training OOF predictions, so the tuned figure is still a training estimate, not an unbiased final score. On the held-out student test set, the tuned pipeline (tuned classifier with tuned HMM parameters) improves all 10 trips, with per-trip gains between +1.1 and +8.5 percentage points; observation-weighted, top-1 moves from 0.8793 (argmax baseline) to 0.9299, a gain of +5.1 points. These student-test figures were produced on the deployment runtime (scikit-learn 1.7.2) using the 1.7.2-native re-save of the tuned classifier, whose probabilities are bit-identical to the original-loaded model on all 155,598 student candidate rows; equivalence to the original scikit-learn 1.9.1 runtime has not been shown and is not claimed. The gain comes from temporal consistency: the decoder trades a small amount of per-observation emission confidence for routes that respect graph connectivity.

### 5.2 Trips where the HMM did not help

The improvement is not uniform. Under default parameters, 77 of 80 training trips improved; three did not:

| trip | observations | baseline top-1 | HMM top-1 | delta |
|------|--------------|----------------|-----------|-------|
| 63 | 2,781 | 0.8713 | 0.8472 | -2.41 pp |
| 72 | 969 | 0.7049 | 0.6821 | -2.27 pp |
| 97 | 1,740 | 0.9856 | 0.9851 | -0.06 pp |

Trip 97 is a tie in practice: the baseline was already near ceiling. Trips 5 and 11 similarly sit above 0.97 baseline, where little headroom remains.

Trips 63 and 72 are the honest regressions. Two plausible mechanisms, stated as hypotheses rather than settled causes: (a) the hop penalty prefers connected, straighter paths, so a genuine short detour or turnaround that the classifier already scored correctly can be smoothed away; (b) trip 72 has the weakest baseline of the three and 152 observations without reference coverage, so local ambiguity gives the transition term more influence than the emissions can support. These regressions were measured under default transition parameters; the committed training artifacts do not include a tuned per-trip breakdown, so we report them as default-parameter behaviour rather than claim the tuning resolves them.

### 5.3 Caveats and scoring notes

- 198 of 208,417 training observations (0.09%) have no candidates at all; the decoder falls back there and no parameter choice can recover them.
- The tuned optimum sits at the corner of all three search grids (lambda, epsilon and gap threshold at their maxima), so a wider grid could improve the result slightly; we kept the verified training-only values for the freeze.
- Continuity scoring matters for interpretation: a whole-trip adjacency count underreports long interrupted runs, while segment-aware scoring (run-collapse, then directed head-to-tail adjacency per segment) reflects what the decoder actually optimizes. We report the segment-aware reading alongside raw top-1 wherever continuity is quoted.
