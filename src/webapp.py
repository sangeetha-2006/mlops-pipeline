"""
Web application for the MLOps pipeline.

One FastAPI app that lets a user run the whole process from a browser:

  - Train:     pick a dataset and augmentation settings, run preprocessing and
               training as a background job, and watch the log live.
  - Registry:  see every registered model version, its stage and its F1-score.
  - Predict:   send a feature vector to the current Production model.
  - Monitor:   request counts and latency of the prediction endpoint.

Training runs the same scripts used everywhere else (preprocess.py, train.py)
in a subprocess, so the web app, the command line, Docker and Kubernetes all
share one implementation of the pipeline.

Run from the src folder:   uvicorn webapp:app --port 8080
"""
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import List

import numpy as np
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from model_store import load_bundle, production_version
from preprocess import DATASET_LOADERS

warnings.filterwarnings("ignore", message="X does not have valid feature names")
logger = logging.getLogger("webapp")

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
STATIC_DIR = SRC_DIR / "static"
# One data folder and one MLflow store for every part of the project.
DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(ROOT_DIR / "data" / "processed")))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or (ROOT_DIR / "mlruns").as_uri()

STAGES = ["preprocess", "augment", "train", "register", "gate"]
STAGE_MARKERS = [
    ("Loading raw data", "preprocess"),
    ("Augmenting training data", "augment"),
    ("] metrics:", "train"),
    ("Registered ", "register"),
    ("Promoted ", "gate"),
    ("Kept existing", "gate"),
]

REQUEST_COUNT = Counter(
    "webapp_prediction_requests_total", "Prediction requests", ["dataset", "status"]
)
REQUEST_LATENCY = Histogram("webapp_prediction_latency_seconds", "Prediction latency in seconds")

app = FastAPI(title="MLOps Control Center", version="1.0.0")

JOBS = {}
JOBS_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()
STATS = {
    "total": 0,
    "errors": 0,
    "latencies_ms": deque(maxlen=500),
    "by_dataset": defaultdict(int),
    "recent": deque(maxlen=10),
}
_BUNDLES = {}
_CHECKED_AT = {}
_META = {}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def registered_name(dataset: str) -> str:
    """Name under which a dataset's model is registered (same rule as train.py)."""
    return f"{dataset}-classifier-prod"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def _check_dataset(dataset: str) -> str:
    dataset = (dataset or "").lower()
    if dataset not in DATASET_LOADERS:
        raise HTTPException(404, f"Unknown dataset '{dataset}'. Available: {list(DATASET_LOADERS)}")
    return dataset


def parse_result_line(line: str):
    """Read the summary line printed at the end of train.py, or return None."""
    prefix = "RESULT_JSON:"
    if not line.startswith(prefix):
        return None
    try:
        return json.loads(line[len(prefix):])
    except ValueError:
        return None


def detect_stage(line: str, current: str) -> str:
    """Move the job's stage forward when a log line marks a later pipeline stage."""
    for marker, stage in STAGE_MARKERS:
        if marker in line and STAGES.index(stage) > STAGES.index(current):
            return stage
    return current


def dataset_meta(name: str) -> dict:
    """Feature names, class names, value ranges and one real example per class."""
    if name not in _META:
        ds = DATASET_LOADERS[name](as_frame=True)
        frame = ds.frame
        features = list(ds.feature_names)
        targets = [str(t) for t in ds.target_names]
        examples = []
        for cls, label in enumerate(targets):
            row = frame[frame["target"] == cls].iloc[0]
            examples.append({"label": label, "values": [round(float(row[f]), 4) for f in features]})
        stats = {
            f: {
                "min": round(float(frame[f].min()), 4),
                "max": round(float(frame[f].max()), 4),
                "mean": round(float(frame[f].mean()), 4),
            }
            for f in features
        }
        _META[name] = {
            "name": name,
            "features": features,
            "n_features": len(features),
            "targets": targets,
            "examples": examples,
            "stats": stats,
            "samples": int(len(frame)),
        }
    return _META[name]


def list_versions(dataset: str) -> list:
    client = _client()
    rows = []
    versions = client.search_model_versions(f"name='{registered_name(dataset)}'")
    for v in sorted(versions, key=lambda v: int(v.version), reverse=True):
        run = client.get_run(v.run_id)
        rows.append({
            "version": int(v.version),
            "stage": v.current_stage,
            "model_type": run.data.params.get("model_type"),
            "f1": run.data.metrics.get("f1_weighted"),
            "accuracy": run.data.metrics.get("accuracy"),
            "augment_factor": run.data.params.get("augment_factor"),
            "created": v.creation_timestamp / 1000,
            "run_id": v.run_id[:8],
        })
    return rows


def get_bundle(dataset: str) -> dict:
    """Return the current Production model for a dataset, reloading it after a promotion."""
    name = registered_name(dataset)
    cached = _BUNDLES.get(dataset)
    if cached and time.time() - _CHECKED_AT.get(dataset, 0) < 2:
        return cached
    client = _client()
    mv = production_version(client, name)
    if mv is None:
        raise HTTPException(
            409, f"No Production model for '{dataset}' yet. Train one on the Train page first."
        )
    if cached and cached["version"] == int(mv.version):
        _CHECKED_AT[dataset] = time.time()
        return cached
    try:
        bundle = load_bundle(
            name, "Production", str(DATA_DIR / dataset / "scaler.joblib"), client=client
        )
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    run = client.get_run(bundle["run_id"])
    bundle["model_type"] = run.data.params.get("model_type")
    bundle["f1"] = run.data.metrics.get("f1_weighted")
    _BUNDLES[dataset] = bundle
    _CHECKED_AT[dataset] = time.time()
    return bundle


# ----------------------------------------------------------------------------
# Training jobs
# ----------------------------------------------------------------------------
def _handle_line(job: dict, line: str):
    result = parse_result_line(line)
    if result is not None:
        job["result"] = result
        return
    if not line.strip():
        return
    job["log"].append(line)
    if len(job["log"]) > 400:
        del job["log"][:100]
    job["stage"] = detect_stage(line, job["stage"])


def _run_job(job: dict):
    env = os.environ.copy()
    env.update({
        "DATASET_NAME": job["dataset"],
        "MLFLOW_TRACKING_URI": TRACKING_URI,
        "PROCESSED_DATA_DIR": str(DATA_DIR),
        "AUGMENT_FACTOR": str(job["augment_factor"]),
        "NOISE_STD": str(job["noise_std"]),
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore",
    })
    # Use each dataset's default experiment and model names.
    env.pop("MLFLOW_EXPERIMENT_NAME", None)
    env.pop("REGISTERED_MODEL_NAME", None)

    job["status"] = "running"
    failed = None
    try:
        for script in ("preprocess.py", "train.py"):
            proc = subprocess.Popen(
                [sys.executable, script], cwd=SRC_DIR, env=env, text=True, bufsize=1,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            for raw in proc.stdout:
                _handle_line(job, raw.rstrip())
            code = proc.wait()
            if code != 0:
                failed = f"{script} stopped with exit code {code}. See the log above."
                break
        if not failed and job.get("result") is None:
            failed = "Training finished but returned no result."
    except Exception as e:  # keep the web app alive whatever happens in a job
        failed = f"Could not run the pipeline: {e}"
    job["error"] = failed
    job["status"] = "failed" if failed else "done"
    job["finished"] = time.time()
    _BUNDLES.pop(job["dataset"], None)


def _job_view(job: dict) -> dict:
    view = {k: v for k, v in job.items() if k != "log"}
    view["log"] = job["log"][-300:]
    return view


class TrainRequest(BaseModel):
    dataset: str = "wine"
    augment_factor: int = Field(2, ge=0, le=10)
    noise_std: float = Field(0.03, ge=0.0, le=0.5)


class PredictRequest(BaseModel):
    dataset: str
    features: List[float]


# ----------------------------------------------------------------------------
# Pages and API
# ----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/config")
def config():
    return {"tracking_uri": TRACKING_URI, "data_dir": str(DATA_DIR), "datasets": list(DATASET_LOADERS)}


@app.get("/api/datasets")
def datasets():
    out = []
    for name in DATASET_LOADERS:
        meta = dict(dataset_meta(name))
        meta["processed"] = (DATA_DIR / name / "scaler.joblib").exists()
        out.append(meta)
    return out


@app.get("/api/overview")
def overview():
    client = _client()
    with JOBS_LOCK:
        running = next((j["id"] for j in JOBS.values() if j["status"] in ("queued", "running")), None)
    cards = []
    for name in DATASET_LOADERS:
        versions = list_versions(name)
        prod = production_version(client, registered_name(name))
        current = next((v for v in versions if prod and v["version"] == int(prod.version)), None)
        cards.append({"dataset": name, "versions": len(versions), "production": current})
    return {"datasets": cards, "running_job": running}


@app.get("/api/registry/{dataset}")
def registry(dataset: str):
    dataset = _check_dataset(dataset)
    return {"dataset": dataset, "model_name": registered_name(dataset), "versions": list_versions(dataset)}


@app.post("/api/train")
def train(req: TrainRequest):
    dataset = _check_dataset(req.dataset)
    with JOBS_LOCK:
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            raise HTTPException(409, "A training job is already running. Wait for it to finish.")
        job = {
            "id": uuid.uuid4().hex[:8],
            "dataset": dataset,
            "augment_factor": req.augment_factor,
            "noise_std": req.noise_std,
            "status": "queued",
            "stage": "preprocess",
            "log": [],
            "result": None,
            "error": None,
            "started": time.time(),
            "finished": None,
        }
        JOBS[job["id"]] = job
        for old in sorted(JOBS.values(), key=lambda j: j["started"])[:-20]:
            JOBS.pop(old["id"], None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job["id"]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return _job_view(job)


@app.post("/api/predict")
def predict(req: PredictRequest):
    dataset = _check_dataset(req.dataset)
    meta = dataset_meta(dataset)
    started = time.perf_counter()
    try:
        if len(req.features) != meta["n_features"]:
            raise HTTPException(
                422, f"'{dataset}' needs {meta['n_features']} values, but {len(req.features)} were sent."
            )
        if not all(math.isfinite(x) for x in req.features):
            raise HTTPException(422, "Every value must be a finite number.")
        bundle = get_bundle(dataset)
        X = bundle["scaler"].transform(np.array(req.features, dtype=float).reshape(1, -1))
        proba = bundle["model"].predict_proba(X)[0]
        classes = [int(c) for c in bundle["model"].classes_]
        best = int(np.argmax(proba))
        notes = []
        for name, value in zip(meta["features"], req.features):
            st = meta["stats"][name]
            margin = 0.5 * (st["max"] - st["min"])
            if value < st["min"] - margin or value > st["max"] + margin:
                notes.append(f"{name} = {value} is far outside the range seen in the data "
                             f"({st['min']} to {st['max']}).")
        result = {
            "dataset": dataset,
            "prediction": classes[best],
            "label": meta["targets"][classes[best]],
            "probabilities": [
                {"label": meta["targets"][c], "probability": round(float(p), 4)}
                for c, p in zip(classes, proba)
            ],
            "model_version": bundle["version"],
            "model_type": bundle.get("model_type"),
            "notes": notes,
        }
    except HTTPException:
        _record(dataset, started, None, ok=False)
        raise
    except Exception as e:
        _record(dataset, started, None, ok=False)
        raise HTTPException(500, f"Prediction failed: {e}")
    _record(dataset, started, result, ok=True)
    return result


def _record(dataset: str, started: float, result, ok: bool):
    elapsed = time.perf_counter() - started
    REQUEST_COUNT.labels(dataset=dataset, status="success" if ok else "error").inc()
    REQUEST_LATENCY.observe(elapsed)
    with STATS_LOCK:
        STATS["total"] += 1
        if not ok:
            STATS["errors"] += 1
            return
        STATS["by_dataset"][dataset] += 1
        STATS["latencies_ms"].append(elapsed * 1000)
        STATS["recent"].appendleft({
            "time": time.time(),
            "dataset": dataset,
            "label": result["label"],
            "confidence": max(p["probability"] for p in result["probabilities"]),
            "model_version": result["model_version"],
        })


@app.get("/api/stats")
def stats():
    with STATS_LOCK:
        lat = sorted(STATS["latencies_ms"])
        return {
            "total": STATS["total"],
            "errors": STATS["errors"],
            "avg_ms": round(sum(lat) / len(lat), 2) if lat else None,
            "p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else None,
            "by_dataset": dict(STATS["by_dataset"]),
            "recent": list(STATS["recent"]),
        }
