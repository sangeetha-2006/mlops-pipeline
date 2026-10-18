# MLOps Pipeline for Augmented Model Deployment

An end-to-end, cloud-agnostic MLOps pipeline: data preprocessing, augmentation,
experiment tracking, model registry with automated promotion gating,
containerized serving, CI/CD, and Kubernetes deployment with autoscaling
and monitoring.

Stack: **MLflow · scikit-learn · FastAPI · Docker · Kubernetes · GitHub Actions · Prometheus**

## Why "augmented"

Two things are "augmented" here, deliberately:

1. **Data augmentation** — the training stage generates jittered synthetic
   copies of the training set (`src/train.py::augment`) so the model sees a
   wider distribution before it ever reaches production.
2. **Augmented decision-making in deployment** — a new model is never blindly
   shipped. It's registered to *Staging*, evaluated against whatever is
   currently in *Production*, and only promoted if it wins.

## Multiple datasets / models

The pipeline is dataset-agnostic via the `DATASET_NAME` environment variable.
Two datasets are wired in out of the box:

| `DATASET_NAME` | Dataset | Features | Registered model name |
|---|---|---|---|
| `wine` (default) | sklearn wine classification | 13 | `wine-classifier-prod` |
| `iris` | sklearn iris classification | 4 | `iris-classifier-prod` |

Each dataset gets its own MLflow experiment, its own registered model, and
its own processed-data folder (`data/processed/<dataset>/`), so training or
promoting one never touches the other. To add a third dataset, add an entry
to `DATASET_LOADERS` in `src/preprocess.py` — everything else (training,
registry gating, serving) picks it up automatically through the same env var.

```bash
# Train and register the wine model (default)
DATASET_NAME=wine python src/preprocess.py && DATASET_NAME=wine python src/train.py

# Train and register the iris model — a separate, independent registration
DATASET_NAME=iris python src/preprocess.py && DATASET_NAME=iris python src/train.py

# Serve either one (run on different ports to serve both at once)
DATASET_NAME=wine uvicorn serve:app --port 8000
DATASET_NAME=iris uvicorn serve:app --port 8001
```

## Web application

A browser interface for the whole process. It runs the same `preprocess.py` and `train.py` used by the command
line, Docker and Kubernetes, so there is one implementation of the pipeline.

```bash
make web            # then open http://localhost:8080
```

| Page | What it does |
|---|---|
| Overview | Production model, version and F1-score for each dataset |
| Train | Choose a dataset and augmentation settings, run the pipeline, watch the live log and the promotion-gate decision |
| Predict | Enter feature values (or fill a real example) and get a prediction with class probabilities |
| Registry | Every registered version with its stage, model type, F1-score and augmentation setting |
| Monitor | Request count, error count and latency of the prediction endpoint; Prometheus metrics at `/metrics` |

The web app uses `./mlruns` as its MLflow store and `./data/processed` for processed data unless
`MLFLOW_TRACKING_URI` or `PROCESSED_DATA_DIR` are set. With Docker Compose it is an optional profile:
`docker compose --profile web up --build`, then open http://localhost:8080.

Limits of this version: there is no login, only one training job runs at a time, job history and request
statistics are kept in memory and reset when the app restarts, and it is meant for a single local user.

### The scaler is stored with the model

Training saves the fitted scaler inside each MLflow run next to the model. Serving and the web app load both
from the registry, so a model can never be served without the scaling it was trained with, including on
Kubernetes where the serving pod cannot see the training pod's files. A model trained before this change has
no stored scaler. Train it once more from the Train page (or `make train`) and it will.

## Project layout

```
mlops-pipeline/
├── src/
│   ├── preprocess.py     # Load, clean, split, scale data (dataset-aware)
│   ├── train.py          # Augment, train candidates, track, register, promote
│   ├── serve.py           # FastAPI serving layer (loads Production model)
│   ├── model_store.py     # Loads a Production model together with its scaler
│   ├── webapp.py          # Web application backend (train, predict, registry, monitor)
│   └── static/index.html  # Web application interface
├── tests/
│   └── test_pipeline.py   # Unit tests run in CI before any image is built
├── k8s/                    # Kubernetes manifests (namespace, MLflow, jobs, serving, ingress, cron)
├── monitoring/
│   └── prometheus.yml      # Scrape config for the serving API
├── .github/workflows/
│   └── ci-cd.yml           # test -> build -> train -> deploy pipeline
├── Dockerfile               # Multi-stage: shared base -> train image -> serve image
├── docker-compose.yml        # One-command local stack (MLflow + train + serve + Prometheus)
├── Makefile                   # make install / test / train / serve / up / down
└── requirements.txt
```

## Quickstart (local)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "setuptools<70"
pip install -r requirements.txt

export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
make preprocess
make train
make serve                  # API on http://localhost:8000
```

Try a prediction once the API is up:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}'
```

## Deploying to Kubernetes

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-mlflow.yaml
kubectl apply -f k8s/02-training-job.yaml
kubectl wait --for=condition=complete job/model-training -n mlops --timeout=600s
kubectl apply -f k8s/03-serving-deployment.yaml
kubectl apply -f k8s/04-ingress.yaml
kubectl apply -f k8s/05-retrain-cronjob.yaml
```

The Job and Deployment manifests reference locally built images
(`mlops-pipeline-train:latest`, `mlops-pipeline-serve:latest`) with
`imagePullPolicy: IfNotPresent`, which works directly with Docker Desktop's
built-in Kubernetes (it shares the local image cache). For a different
cluster, build/push to a real registry and update the `image:` fields.

## CI/CD (`.github/workflows/ci-cd.yml`)

`test → build-and-push → train-and-register → deploy`, each gated on the
previous stage succeeding. Requires a `KUBECONFIG_DATA` repo secret
(base64-encoded kubeconfig) for the cluster-dependent stages.

## Observability

- `/health` — liveness/readiness
- `/metrics` — Prometheus format: request counts by status, latency histogram
- `/reload` — hot-swap to the latest Production model version without a pod restart

See `ARCHITECTURE.md` for a deeper dive into pipeline design and decisions.
