import time
from contextlib import asynccontextmanager
from pathlib import Path

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("models") / "boosting.ubj"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError(
            f"Model file {MODEL_PATH} not found. Please run train.py first."
        )
    yield


app = FastAPI(
    title="Logistic Regression Model API",
    description="A simple API for serving logistic regression predictions.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    hour: int
    weekday: int
    month: int
    amount: float


class PredictionResponse(BaseModel):
    probability: list[float]
    inference_time: float


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """Accepts feature arrays, runs inference, and returns predicted class and probabilities."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    features = [[payload.hour, payload.weekday, payload.month, payload.amount]]

    start = time.perf_counter()
    probabilities = model.predict_proba(features)[0].tolist()
    inference_time = time.perf_counter() - start

    return PredictionResponse(
        probability=probabilities, inference_time=round(inference_time, 6)
    )
