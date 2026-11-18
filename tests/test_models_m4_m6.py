from sklearn.pipeline import Pipeline

from training.harness import fit_cv, make_pipeline
from training.mock_data import make_mock
from training.models_m4_m6 import build_models, get_model


def test_m4_m6_have_expected_settings_and_probabilities():
    models = build_models()

    assert set(models) == {"knn", "hist_gradient_boosting", "mlp"}
    for spec in models.values():
        assert hasattr(spec["model"], "predict_proba")
        assert spec["row_cap"] > 0
        assert isinstance(make_pipeline(spec["model"], scale=spec["scale"]), Pipeline)

    assert models["knn"]["scale"] is True
    assert models["mlp"]["scale"] is True
    assert models["hist_gradient_boosting"]["scale"] is False


def test_each_model_runs_five_grouped_folds_on_mock_data():
    df = make_mock(n_trips=10, rows_per_trip=40)

    for spec in build_models().values():
        rows = fit_cv(spec["model"], df, scale=spec["scale"], row_cap=spec["row_cap"])
        assert len(rows) == 5
        assert all("macro_f1" in row and "log_loss" in row for row in rows)


def test_get_model_rejects_unknown_name():
    try:
        get_model("not_a_model")
    except KeyError as exc:
        assert "Unknown model" in str(exc)
    else:
        raise AssertionError("unknown model should raise KeyError")
