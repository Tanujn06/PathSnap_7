from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.data_io import load_trip
from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, decode
from training.models_m4_m6 import get_model
from training.harness import (
    SEED, ROW_CAPS, split_xy, make_pipeline, row_sample
)

DATA = Path("data_cache/features_train.parquet")
RAW = Path("Student folder/train")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)
x, y, groups, cols = split_xy(df)
oof = np.full(len(df), np.nan, dtype=np.float64)

cv = GroupKFold(n_splits=5)
spec = get_model("hist_gradient_boosting")
print("Starting grouped 5-fold HGB OOF training...", flush=True)

for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y, groups), 1):
    start = time.time()
    train = df.iloc[tr_idx]
    valid = df.iloc[va_idx]

    train = row_sample(
        train, ROW_CAPS["hist_gradient_boosting"], SEED + fold
    )
    model = make_pipeline(spec["model"], scale=False)
    model.fit(train[cols], train["label"].astype(int))
    oof[va_idx] = model.predict_proba(valid[cols])[:, 1]

    print(
        f"Fold {fold}/5 complete | train={len(train):,} "
        f"valid={len(valid):,} | {time.time()-start:.1f}s",
        flush=True
    )

if not np.isfinite(oof).all():
    raise RuntimeError("OOF predictions contain missing/non-finite values")

prob = df[["trip_id", "obs_idx", "arc_id"]].copy()
prob["p_match"] = np.clip(oof, 1e-9, 1.0)
prob.to_parquet(OUT / "hgb_train_oof.parquet", index=False)

scored = df[
    ["trip_id", "obs_idx", "arc_id", "label"]
].copy()
scored["p_match"] = oof

params = HMMParams()
results = []
all_predictions = []

def trip_dir(tid):
    p = RAW / tid
    if not p.is_dir():
        raise FileNotFoundError(f"Missing raw training trip: {p}")
    return p

print("\nEvaluating OOF HGB and HGB + HMM by trip...", flush=True)

for tid, group in scored.groupby("trip_id", sort=True):
    trip = load_trip(trip_dir(tid))
    graph = RoadGraph(trip.nodes, trip.arcs)

    group = group.sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
    emissions = np.log(np.clip(group["p_match"].to_numpy(), 1e-9, 1.0))

    times = trip.track[:, 2].astype(np.float64)
    times = times - times[0]

    decoded = decode(
        group[["obs_idx", "arc_id"]],
        emissions,
        graph,
        params=params,
        obs_times=times,
        n_obs=len(trip.track),
    )

    truth = (
        group[group.label == 1]
        .drop_duplicates("obs_idx")
        .set_index("obs_idx")["arc_id"]
    )

    baseline = (
        group.sort_values(
            ["obs_idx", "p_match", "arc_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("obs_idx")
        .set_index("obs_idx")["arc_id"]
    )

    obs = np.arange(len(trip.track))
    true_arc = truth.reindex(obs)
    baseline_arc = baseline.reindex(obs)

    hmm_ok = pd.Series(decoded, index=obs).eq(true_arc) & true_arc.notna()
    base_ok = baseline_arc.eq(true_arc) & true_arc.notna()

    results.append({
        "trip_id": tid,
        "observations": len(obs),
        "known_truth_observations": int(true_arc.notna().sum()),
        "baseline_top1": float(base_ok.sum() / len(obs)),
        "hmm_top1": float(hmm_ok.sum() / len(obs)),
    })

    all_predictions.extend({
        "trip_id": tid,
        "obs_idx": int(i),
        "pred_arc": int(decoded[i]),
    } for i in obs)

report = pd.DataFrame(results)
n = report.observations.sum()
baseline_score = np.average(report.baseline_top1, weights=report.observations)
hmm_score = np.average(report.hmm_top1, weights=report.observations)

print("\n========== TRAINING OOF VALIDATION ==========")
print(report.to_string(index=False, formatters={
    "baseline_top1": "{:.4%}".format,
    "hmm_top1": "{:.4%}".format,
}))
print(f"\nOOF HGB:     {baseline_score:.4%}")
print(f"OOF HGB+HMM: {hmm_score:.4%}")
print(f"Delta:       {(hmm_score-baseline_score)*100:+.2f} pp")
print(f"Observations: {n:,}")
print(f"Truth coverage: {report.known_truth_observations.sum()/n:.4%}")

report.to_csv(OUT / "hmm_train_oof_report.csv", index=False)
pd.DataFrame(all_predictions).to_parquet(
    OUT / "hmm_train_oof_predictions.parquet", index=False
)

summary = {
    "method": "5-fold GroupKFold OOF",
    "model": "HistGradientBoostingClassifier",
    "hmm_params": vars(params),
    "baseline_top1": float(baseline_score),
    "hmm_top1": float(hmm_score),
    "delta_percentage_points": float((hmm_score-baseline_score)*100),
    "n_observations": int(n),
    "truth_coverage": float(report.known_truth_observations.sum()/n),
}
(OUT / "hmm_train_oof_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("\nSaved OOF probabilities, predictions, report and summary.")
