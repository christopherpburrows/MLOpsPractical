import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
EDUCATIONAL_NOTICE = "Historical census-area estimate for teaching, not a current property appraisal."


class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    median_income: float = Field(gt=0, description="Area median income in units of $10,000")
    house_age: float = Field(gt=0, description="Area median house age in years")
    average_rooms: float = Field(gt=0, description="Average rooms per household")
    average_bedrooms: float = Field(gt=0, description="Average bedrooms per household")
    population: float = Field(gt=0, description="Population of the census block group")
    average_occupancy: float = Field(gt=0, description="Average people per household")
    latitude: float = Field(ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(ge=-180, le=180, description="Longitude in degrees")


class Prediction(BaseModel):
    predicted_median_house_value_usd: float
    model_version: str
    notice: str


def create_app(model_path: Path | None = None) -> FastAPI:
    artifact_path = model_path if model_path is not None else Path(
        os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not artifact_path.is_file():
            raise RuntimeError(f"Model not found at {artifact_path}. Run python train.py first.")
        bundle = joblib.load(artifact_path)
        if bundle["metadata"]["sklearn_version"] != sklearn.__version__:
            raise RuntimeError("Training and serving must use the same scikit-learn version.")
        if bundle["metadata"]["feature_keys"] != list(Measurements.model_fields):
            raise RuntimeError("Model features do not match the housing API. Run python train.py again.")
        application.state.bundle = bundle
        yield
        application.state.bundle = None

    application = FastAPI(
        title="California Housing Value API",
        description=EDUCATIONAL_NOTICE,
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["Operations"])
    def health() -> dict:
        return {"status": "alive"}

    @application.get("/ready", tags=["Operations"], responses={503: {"description": "Model is not loaded"}})
    def ready(request: Request) -> dict:
        if getattr(request.app.state, "bundle", None) is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ready"}

    @application.get("/model", tags=["Operations"])
    def model_info(request: Request) -> dict:
        return request.app.state.bundle["metadata"]

    @application.post("/predict", tags=["Inference"])
    def predict(measurements: Measurements, request: Request) -> Prediction:
        bundle = request.app.state.bundle
        model = bundle["model"]
        metadata = bundle["metadata"]
        values = measurements.model_dump()
        features = [[values[feature] for feature in metadata["feature_keys"]]]
        return Prediction(
            predicted_median_house_value_usd=round(float(model.predict(features)[0]), 2),
            model_version=metadata["model_version"],
            notice=EDUCATIONAL_NOTICE,
        )

    return application


app = create_app()