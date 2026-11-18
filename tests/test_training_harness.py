import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from training.harness import fit_cv, load_features, make_pipeline, split_xy
from training.mock_data import make_mock


def test_mock_table_runs_group_cv():
    df = make_mock(n_trips=10, rows_per_trip=20)
    folds = fit_cv(LogisticRegression(max_iter=200), df, scale=True)
    assert len(folds) == 5
    assert all("macro_f1" in row and "log_loss" in row for row in folds)


def test_loader_rejects_missing_contract_columns(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"trip_id": ["t1"], "label": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing columns"):
        load_features(path)


def test_split_keeps_ids_out_of_features():
    df = make_mock(n_trips=5, rows_per_trip=3)
    x, y, groups, cols = split_xy(df)
    assert "trip_id" not in cols and "arc_id" not in cols
    assert len(x) == len(y) == len(groups)


def test_pipeline_contains_fold_local_imputation_and_scaling():
    pipe = make_pipeline(LogisticRegression(), scale=True)
    assert [name for name, _ in pipe.steps] == ["impute", "scale", "model"]
