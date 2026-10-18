"""
Model-serving layer: a FastAPI app that loads the current "Production"
model version from the MLflow Model Registry and exposes it as a REST
API. Designed to run as a standalone container behind Docker/K8s.

Dataset-aware: which model gets served is entirely controlled by env
vars (REGISTERED_MODEL_NAME, SCALER_PATH, N_FEATURES), so the exact
same image serves "wine" or "iris" depending on how it's configured —
run two instances side by side (different ports) to serve both models
at once.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import List

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from model_store import load_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv("DATASET_NAME", "wine").lower()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{DATASET_NAME}-classifier-prod")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
SCALER_PATH = os.getenv("SCALER_PATH", f"data/processed/{DATASET_NAME}/scaler.joblib")
# wine has 13 features, iris has 4 — default assumes wine unless overridden
N_FEATURES = int(os.getenv("N_FEATURES", 13 if DATASET_NAME == "wine" else 4))

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

model_state = {"model": None, "scaler": None, "version": None}


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Loading %s (%s) from the registry", REGISTERED_MODEL_NAME, MODEL_STAGE)
    # Loads the model and the scaler stored with its MLflow run. Fails loudly if the
    # scaler is missing, instead of silently serving unscaled predictions.
    bundle = load_bundle(REGISTERED_MODEL_NAME, MODEL_STAGE, SCALER_PATH)
    model_state["model"] = bundle["model"]
    model_state["scaler"] = bundle["scaler"]
    model_state["version"] = bundle["version"]
    logger.info("Model v%s loaded successfully. Serving dataset: %s", bundle["version"], DATASET_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            load_model()
            break
        except Exception as e:
            logger.error("Model load attempt %d/%d failed: %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                await asyncio.sleep(5 * attempt)  # 5s, 10s, 15s, 20s backoff
            else:
                logger.error("Giving up on startup model load; use POST /reload once a model is available.")
    yield


app = FastAPI(title=f"MLOps Model Serving API ({DATASET_NAME})", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Feature vector of length {N_FEATURES}",
    )


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int
    probabilities: List[float]
    model_stage: str
    dataset: str = DATASET_NAME


@app.get("/health")
def health():
    ready = model_state["model"] is not None
    return {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "dataset": DATASET_NAME,
        "version": model_state["version"],
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model():
    """Hot-reload the latest Production model without restarting the pod."""
    load_model()
    return {"status": "reloaded", "dataset": DATASET_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model_state["model"] is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    with REQUEST_LATENCY.time():
        try:
            X = np.array(req.features).reshape(1, -1)
            if model_state["scaler"] is not None:
                X = model_state["scaler"].transform(X)

            pred = model_state["model"].predict(X)[0]
            proba = model_state["model"].predict_proba(X)[0].tolist()

            REQUEST_COUNT.labels(status="success").inc()
            return PredictionResponse(
                prediction=int(pred), probabilities=proba, model_stage=MODEL_STAGE, dataset=DATASET_NAME
            )
        except Exception as e:
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
