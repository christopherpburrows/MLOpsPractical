import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURE_KEYS = [
    "median_income", "house_age", "average_rooms", "average_bedrooms",
    "population", "average_occupancy", "latitude", "longitude",
]
RANDOM_STATE = 42


def export_rows(path: Path, features, prices) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([*FEATURE_KEYS, "median_house_value_usd"])
        for measurements, price in zip(features, prices, strict=True):
            writer.writerow([*measurements.tolist(), float(price)])


def train_model(output_dir: Path = ARTIFACT_DIR) -> dict:
    dataset = fetch_california_housing()
    prices_usd = dataset.target * 100_000
    features_train, features_test, prices_train, prices_test = train_test_split(
        dataset.data, prices_usd, test_size=0.2, random_state=RANDOM_STATE,
    )
    candidates = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ], memory=None),
        "random_forest": Pipeline([
            ("regressor", RandomForestRegressor(
                n_estimators=50, max_depth=16, random_state=RANDOM_STATE, n_jobs=1,
                min_samples_leaf=1, max_features=1.0,
            )),
        ], memory=None),
        "histgb": Pipeline([
            ("regressor", HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.1, random_state=RANDOM_STATE,
            )),
        ], memory=None),
    }
    folds = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for name, candidate in candidates.items():
        cv_mae = -cross_val_score(
            candidate, features_train, prices_train, cv=folds, scoring="neg_mean_absolute_error",
        )
        comparison[name] = {
            "cv_mae_mean_usd": float(cv_mae.mean()),
            "cv_mae_std_usd": float(cv_mae.std()),
            "cv_fold_mae_usd": cv_mae.tolist(),
        }
    selected_name = min(comparison, key=lambda name: comparison[name]["cv_mae_mean_usd"])
    for name, candidate in candidates.items():
        candidate.fit(features_train, prices_train)
        test_predictions = candidate.predict(features_test)
        comparison[name].update({
            "test_mae_usd": float(mean_absolute_error(prices_test, test_predictions)),
            "test_rmse_usd": float(root_mean_squared_error(prices_test, test_predictions)),
            "test_r2": float(r2_score(prices_test, test_predictions)),
        })
    model = candidates[selected_name]
    predictions = model.predict(features_test)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(features_train, prices_train)
    metrics = {
        **comparison[selected_name],
        "selected_model": selected_name,
        "selection_metric": "lowest training-only cross-validation MAE",
        "model_comparison": comparison,
        "baseline_mae_usd": float(mean_absolute_error(prices_test, baseline.predict(features_test))),
        "train_samples": len(prices_train),
        "test_samples": len(prices_test),
    }
    metadata = {
        "model_version": f"housing-{selected_name}-v1",
        "model_name": selected_name,
        "selection_metric": "lowest training-only cross-validation MAE",
        "dataset": "California Housing, 1990 US Census, via scikit-learn",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "feature_keys": FEATURE_KEYS,
        "dataset_feature_names": dataset.feature_names,
        "target": "median_house_value_usd",
        "record_unit": "census block group, not an individual house",
        "intended_use": "Teaching historical area-level values, not current property appraisals.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, output_dir / "model.joblib")
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for name, candidate in candidates.items():
        candidate_metadata = {
            **metadata, "model_name": name, "model_version": f"housing-{name}-v1",
        }
        joblib.dump({"model": candidate, "metadata": candidate_metadata}, models_dir / f"{name}.joblib")
    with (output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "model", "selected", "cv_mae_mean_usd", "test_mae_usd", "test_rmse_usd", "test_r2",
        ])
        writer.writeheader()
        for name, result in comparison.items():
            writer.writerow({
                "model": name, "selected": name == selected_name,
                **{key: result[key] for key in writer.fieldnames[2:]},
            })
    export_rows(output_dir / "training_data.csv", features_train, prices_train)
    export_rows(output_dir / "test_data.csv", features_test, prices_test)
    examples = [
        {"actual_usd": float(actual), "predicted_usd": float(predicted),
         "absolute_error_usd": float(abs(actual - predicted))}
        for actual, predicted in zip(prices_test[:5], predictions[:5], strict=True)
    ]
    for filename, content in [
        ("metrics.json", metrics), ("metadata.json", metadata),
        ("prediction_examples.json", examples),
        ("sample_request.json", dict(zip(FEATURE_KEYS, features_test[0].tolist(), strict=True))),
    ]:
        (output_dir / filename).write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Selected {selected_name} using training CV; artifacts saved to {output_dir.resolve()}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a real-data house-value regressor.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    arguments = parser.parse_args()
    train_model(arguments.output_dir)