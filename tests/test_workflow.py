import csv
import json

import joblib
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import Measurements, create_app
from smoke_test import SAMPLE
from train import FEATURE_KEYS, train_model


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("artifacts")
    train_model(output_dir)
    return output_dir


@pytest.fixture
def client(artifacts):
    with TestClient(create_app(artifacts / "model.joblib")) as test_client:
        yield test_client


def test_training_and_saved_pipeline(artifacts):
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["train_samples"] == 16512
    assert metrics["test_samples"] == 4128
    assert 0 < metrics["test_mae_usd"] < metrics["baseline_mae_usd"]
    assert metrics["test_r2"] > 0.75
    assert len(metrics["cv_fold_mae_usd"]) == 3
    for filename, count in [("training_data.csv", 16512), ("test_data.csv", 4128)]:
        with (artifacts / filename).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            assert reader.fieldnames == [*FEATURE_KEYS, "median_house_value_usd"]
            assert sum(1 for row in reader) == count
    bundle = joblib.load(artifacts / "model.joblib")
    assert list(bundle["model"].named_steps)[-1] == "regressor"
    assert bundle["metadata"]["feature_keys"] == FEATURE_KEYS


def test_prediction_matches_saved_model(client, artifacts):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    prediction = response.json()
    assert prediction["predicted_median_house_value_usd"] > 0
    assert prediction["model_version"] == client.get("/model").json()["model_version"]
    assert "not a current property appraisal" in prediction["notice"]
    model = joblib.load(artifacts / "model.joblib")["model"]
    expected = model.predict([[SAMPLE[key] for key in FEATURE_KEYS]])[0]
    assert prediction["predicted_median_house_value_usd"] == pytest.approx(expected, abs=0.01)
    reordered = dict(reversed(list(SAMPLE.items())))
    assert client.post("/predict", json=reordered).json() == prediction


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**SAMPLE, "median_income": -1},
        {**SAMPLE, "population": 0},
        {**SAMPLE, "house_age": "41"},
        {**SAMPLE, "average_rooms": True},
        {**SAMPLE, "latitude": 91},
        {**SAMPLE, "longitude": -181},
        {**SAMPLE, "unexpected": 1},
    ],
)
def test_invalid_measurements_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_are_rejected(value):
    with pytest.raises(ValidationError):
        Measurements(**{**SAMPLE, "median_income": value})


def test_operations_and_documentation(client):
    assert client.get("/health").json() == {"status": "alive"}
    assert client.get("/ready").json() == {"status": "ready"}
    assert client.get("/model").json()["feature_keys"] == FEATURE_KEYS
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_missing_model_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Run python train.py first"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass


def test_incompatible_model_version_fails_at_startup(artifacts, tmp_path):
    bundle = joblib.load(artifacts / "model.joblib")
    bundle["metadata"]["sklearn_version"] = "incompatible"
    model_path = tmp_path / "incompatible.joblib"
    joblib.dump(bundle, model_path)
    with pytest.raises(RuntimeError, match="same scikit-learn version"):
        with TestClient(create_app(model_path)):
            pass


def test_exported_request_matches_first_test_row(client, artifacts):
    request = json.loads((artifacts / "sample_request.json").read_text(encoding="utf-8"))
    examples = json.loads((artifacts / "prediction_examples.json").read_text(encoding="utf-8"))
    result = client.post("/predict", json=request).json()
    assert result["predicted_median_house_value_usd"] == pytest.approx(examples[0]["predicted_usd"], abs=0.01)


def test_three_models_and_training_only_selection(artifacts):
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    comparison = metrics["model_comparison"]
    assert set(comparison) == {"linear_regression", "random_forest", "histgb"}
    best_name = min(comparison, key=lambda name: comparison[name]["cv_mae_mean_usd"])
    assert metrics["selected_model"] == best_name
    deployed = joblib.load(artifacts / "model.joblib")
    assert deployed["metadata"]["model_name"] == best_name
    for name, result in comparison.items():
        assert len(result["cv_fold_mae_usd"]) == 3
        assert result["test_mae_usd"] > 0
        candidate = joblib.load(artifacts / "models" / f"{name}.joblib")
        assert candidate["metadata"]["model_name"] == name
        assert candidate["model"].n_features_in_ == len(FEATURE_KEYS)
    with (artifacts / "model_comparison.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 3
    assert [row["model"] for row in rows if row["selected"] == "True"] == [best_name]