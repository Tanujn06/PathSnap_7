from pathlib import Path
from itertools import product
import json
import time

import numpy as np
import pandas as pd

from pathsnap.data_io import load_trip
from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, decode

FEATURES = Path("results/hgb_train_oof.parquet")
RAW = Path("Student folder/train")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(FEATURES)

# OOF predictions may omit labels; recover them from the source feature table.
if "label" not in df.columns:
    source = pd.read_parquet("data_cache/features_train.parquet")
    keys = ["trip_id", "obs_idx", "arc_id"]
    labels = source[keys + ["label"]].drop_duplicates(keys)
    df = df.merge(labels, on=keys, how="left", validate="one_to_one")
    if df["label"].isna().any():
        raise RuntimeError("Some OOF candidates could not be matched to training labels.")
    print("Recovered ground-truth labels from features_train.parquet.")

required = {"trip_id", "obs_idx", "arc_id", "label", "p_match"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns in OOF file: {sorted(missing)}")

# Prepare each trip once; graph transition caches are reused across trials.
trips = []
for tid, cands in df.groupby("trip_id", sort=True):
    tid = str(tid)
    trip = load_trip(RAW / tid)
    cands = cands.copy().sort_values(["obs_idx", "arc_id"])
    n_obs = len(trip.track)

    truth = (
        cands.loc[cands.label == 1]
        .drop_duplicates("obs_idx")
        .set_index("obs_idx")["arc_id"]
    )
    truth = truth.reindex(range(n_obs))

    baseline = (
        cands.sort_values(
            ["obs_idx", "p_match", "arc_id"],
            ascending=[True, False, True]
        )
        .drop_duplicates("obs_idx")
        .set_index("obs_idx")["arc_id"]
        .reindex(range(n_obs))
    )

    trips.append({
        "id": tid,
        "cands": cands,
        "graph": RoadGraph(trip.nodes, trip.arcs),
        "times": np.asarray(trip.track)[:, 2],
        "n": n_obs,
        "truth": truth,
        "baseline": baseline,
    })

print(f"Loaded {len(trips)} training trips.")

# Search only HMM transition parameters. Emission probabilities stay fixed.
grid = list(product(
    [0.5, 1.0, 2.0, 3.0, 4.0],   # lambda_per_hop
    [10.0, 25.0, 50.0, 100.0],   # unreachable-transition penalty
    [3.0, 5.0, 10.0],            # segment gap threshold, seconds
))

results = []
best = None
start = time.time()

for trial, (lam, eps, gap) in enumerate(grid, 1):
    params = HMMParams(
        sigma_m=12.5,
        lambda_per_hop=lam,
        epsilon=eps,
        w=1.0,
        gap_threshold_s=gap,
    )

    correct = 0
    baseline_correct = 0
    total = 0
    known = 0

    for item in trips:
        cands = item["cands"]
        emissions = np.log(
            np.clip(cands["p_match"].to_numpy(dtype=float), 1e-12, 1.0)
        )

        decoded = decode(
            cands,
            emissions,
            item["graph"],
            params,
            obs_times=item["times"],
            n_obs=item["n"],
        )

        truth = item["truth"]
        base = item["baseline"]
        valid = truth.notna().to_numpy()

        pred = pd.Series(decoded).reindex(range(item["n"]))
        hmm_ok = pred.to_numpy() == truth.to_numpy()
        base_ok = base.to_numpy() == truth.to_numpy()

        correct += int((hmm_ok & valid).sum())
        baseline_correct += int((base_ok & valid).sum())
        known += int(valid.sum())
        total += item["n"]

    score = correct / total
    baseline_score = baseline_correct / total
    known_score = correct / known if known else 0.0

    row = {
        "lambda_per_hop": lam,
        "epsilon": eps,
        "gap_threshold_s": gap,
        "hmm_top1": score,
        "baseline_top1": baseline_score,
        "delta_pp": (score - baseline_score) * 100,
        "hmm_accuracy_known_truth": known_score,
        "known_truth_observations": known,
        "total_observations": total,
    }
    results.append(row)

    if best is None or score > best["hmm_top1"]:
        best = row

    if trial % 10 == 0 or trial == len(grid):
        print(
            f"Trial {trial:02}/{len(grid)} | "
            f"best={best['hmm_top1']:.4%} "
            f"(lambda={best['lambda_per_hop']}, "
            f"epsilon={best['epsilon']}, gap={best['gap_threshold_s']})"
        )

report = pd.DataFrame(results).sort_values(
    ["hmm_top1", "hmm_accuracy_known_truth"],
    ascending=False
)
report.to_csv(OUT / "hmm_parameter_search.csv", index=False)

best_params = {
    "note": (
        "Selected using training-only 5-fold GroupKFold OOF predictions. "
        "Do not interpret the best grid score as an unbiased final estimate."
    ),
    "sigma_m": 12.5,
    "lambda_per_hop": best["lambda_per_hop"],
    "epsilon": best["epsilon"],
    "w": 1.0,
    "gap_threshold_s": best["gap_threshold_s"],
}
Path("configs/hmm_params.json").write_text(
    json.dumps(best_params, indent=2) + "\n"
)
(OUT / "hmm_tuning_summary.json").write_text(
    json.dumps({
        "best_parameters": best_params,
        "best_training_oof_metrics": best,
        "baseline_oof_top1": best["baseline_top1"],
        "best_oof_top1": best["hmm_top1"],
        "delta_percentage_points": best["delta_pp"],
        "trials": len(grid),
        "elapsed_seconds": time.time() - start,
    }, indent=2) + "\n"
)

print("\n========== HMM PARAMETER SEARCH ==========")
print(report.head(10).to_string(index=False, formatters={
    "hmm_top1": "{:.4%}".format,
    "baseline_top1": "{:.4%}".format,
    "hmm_accuracy_known_truth": "{:.4%}".format,
    "delta_pp": "{:+.2f}".format,
}))
print("\nSelected configuration:")
print(json.dumps(best_params, indent=2))
print("\nSaved:")
print("  results/hmm_parameter_search.csv")
print("  results/hmm_tuning_summary.json")
print("  configs/hmm_params.json")
print(f"Elapsed: {time.time() - start:.1f}s")
