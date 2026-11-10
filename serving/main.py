from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("models") / "logreg.pk"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
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
    TransactionAmt: float
    C1: int 
    C2: int 
    C3: int 
    C4: int 
    C5: int 
    C6: int 
    C7: int 
    C8: int


class PredictionResponse(BaseModel):
    prediction: int
    probability: list[float]


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """Accepts feature arrays, runs inference, and returns predicted class and probabilities."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        features = [[
            payload.TransactionAmt,
            payload.C1,
            payload.C2,
            payload.C3,
            payload.C4,
            payload.C5,
            payload.C6,
            payload.C7,
            payload.C8,
        ]]

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()

        return PredictionResponse(prediction=prediction, probability=probabilities)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )