import numpy as np
import pandas as pd
import pytest
from training.harness import group_cv, split_xy
from training.run_m1_m3 import top1_accuracy, run_model
from training.models_m1_m3 import build_models


def make_candidate_data():
    rows = []
    rng = np.random.default_rng(42)

    for trip in range(10):
        for obs in range(5):
            for candidate in range(2):
                label = int(candidate == 0)
                rows.append({
                    "trip_id": trip,
                    "obs_idx": obs,
                    "arc_id": candidate,
                    "label": label,
                    "feature_a": float(label + rng.normal(0, 0.1)),
                    "feature_b": float(rng.normal()),
                })

    return pd.DataFrame(rows)


def test_top1_accuracy_ranks_candidates():
    frame = pd.DataFrame({
        "trip_id": [1, 1, 2, 2],
        "obs_idx": [0, 0, 0, 0],
        "label": [1, 0, 1, 0],
    })

    assert top1_accuracy(frame, np.array([0.9, 0.1, 0.8, 0.2])) == 1.0


def test_top1_accuracy_rejects_mismatched_lengths():
    frame = make_candidate_data().head(4)

    with pytest.raises(ValueError, match="lengths"):
        top1_accuracy(frame, np.array([0.5]))


def test_top1_accuracy_rejects_non_finite_probabilities():
    frame = make_candidate_data().head(4)

    with pytest.raises(ValueError, match="finite"):
        top1_accuracy(frame, np.array([0.1, np.nan, 0.3, 0.4]))


def test_run_model_generates_complete_oof_predictions(tmp_path):
    frame = make_candidate_data()
    spec = build_models()["gaussian_nb"]

    records = run_model(
        "gaussian_nb",
        spec,
        frame,
        ["feature_a", "feature_b"],
        tmp_path / "results",
        tmp_path / "probabilities",
    )

    assert len(records) == 5
    assert {record["fold"] for record in records} == {1, 2, 3, 4, 5}

    output = pd.read_parquet(
        tmp_path / "probabilities" / "probs_gaussian_nb_train.parquet"
    )

    assert list(output.columns) == [
        "trip_id", "obs_idx", "arc_id", "p_match"
    ]
    assert len(output) == len(frame)
    assert output["p_match"].notna().all()
    assert output["p_match"].between(0, 1).all()


def test_run_model_rejects_too_few_trips(tmp_path):
    frame = make_candidate_data()
    frame = frame[frame["trip_id"] < 4]

    with pytest.raises(ValueError, match="at least 5"):
        run_model(
            "gaussian_nb",
            build_models()["gaussian_nb"],
            frame,
            ["feature_a", "feature_b"],
            tmp_path / "results",
            tmp_path / "probabilities",
        )

#tests
def test_group_cv_keeps_trips_separate():
    frame = make_candidate_data()
    x, y, groups, _ = split_xy(
        frame,
        ["feature_a", "feature_b"],
    )

    for train_idx, valid_idx in group_cv().split(x, y, groups):
        train_trips = set(groups.iloc[train_idx])
        valid_trips = set(groups.iloc[valid_idx])

        assert train_trips.isdisjoint(valid_trips)
