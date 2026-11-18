from fastapi.testclient import TestClient

from api.main import create_app


class FakePredictor:
    def predict_trip(self, trip_dir):
        return {
            "trip_id": trip_dir.name,
            "n_obs": 2,
            "gps": [[46.0, 53.0], [46.1, 53.1]],
            "predicted_arcs": [10, 11],
            "predicted_path": [[53.0, 46.0], [53.1, 46.1]],
            "predicted_segments": [[[53.0, 46.0], [53.1, 46.1]]],
            "prediction_mode": "model_hmm",
            "mock": False,
            "latency_ms": 5.0,
        }


def make_client(tmp_path):
    trip = tmp_path / "trip-1"
    trip.mkdir()
    (trip / "trip-1.track").write_text(
        "53.0 46.0 1000\n53.1 46.1 1010\n"
    )
    app = create_app(
        data_dir=tmp_path,
        log_path=tmp_path / "test-api.log",
        predictor=FakePredictor(),
    )
    return TestClient(app)


def test_health_reports_model_mode(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["prediction_mode"] == "model_hmm"
    assert response.json()["mock"] is False


def test_trips_lists_valid_trip(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/trips")
    assert response.status_code == 200
    assert "trip-1" in response.json()["trips"]


def test_predict_uses_predictor(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/predict", json={"trip_id": "trip-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction_mode"] == "model_hmm"
    assert body["mock"] is False
    assert body["predicted_arcs"] == [10, 11]
    assert body["predicted_segments"]


def test_predict_unknown_trip_returns_404(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/predict", json={"trip_id": "missing"})
    assert response.status_code == 404


def test_predict_rejects_invalid_trip_id(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/predict", json={"trip_id": "../outside"})
    assert response.status_code == 422
