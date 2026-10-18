# Architecture

## Goals

- **Reproducibility**: every model in production is traceable to a specific
  commit, dataset version, and set of hyperparameters.
- **Safety by default**: no model reaches production without passing an
  automated comparison against the current incumbent.
- **Portability**: no cloud-specific services in the core pipeline.
- **Observability**: every serving replica exposes health and metrics
  endpoints out of the box.
- **Extensibility**: the same code trains, registers, and serves multiple
  independent models (e.g. `wine`, `iris`) purely via the `DATASET_NAME`
  environment variable — no branching logic, no duplicated scripts.

## Pipeline stages

```
 ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
 │   Ingest    │───▶│  Preprocess   │───▶│   Augment +    │───▶│  Model Registry │
 │ (raw data)  │    │ clean/split/  │    │  Train (N      │    │  (per dataset,  │
 │             │    │   scale       │    │  candidates)   │    │  Staging->Prod) │
 └─────────────┘    └──────────────┘    └───────────────┘    └────────┬───────┘
                                                                        │
                                                                        ▼
 ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
 │  Monitoring  │◀───│   Serving     │◀───│   Container    │◀───│  CI/CD deploys │
 │ (Prometheus) │    │  (FastAPI)    │    │   (Docker)     │    │   the image    │
 └─────────────┘    └──────────────┘    └───────────────┘    └────────────────┘
```

### 1. Ingest & preprocess (`src/preprocess.py`)

Loads raw data for whichever dataset `DATASET_NAME` selects (`wine` or
`iris`), deduplicates, drops nulls, splits train/test, and fits a
`StandardScaler`. Output is written to `data/processed/<dataset>/`, keeping
each dataset's artifacts fully isolated.

### 2. Augment & train (`src/train.py`)

- **Augmentation**: Gaussian jitter creates `AUGMENT_FACTOR` noisy copies of
  the training set.
- **Multi-candidate training**: Random Forest, Gradient Boosting, and
  Logistic Regression are trained per dataset, each as its own MLflow run.
- **Per-dataset registration & promotion gate**: the best candidate is
  registered under `<dataset>-classifier-prod` (e.g. `wine-classifier-prod`,
  `iris-classifier-prod`), moved to *Staging*, and compared only against the
  Production version of that same dataset's model — the two models never
  interfere with each other's promotion history.

### 3. Serve (`src/serve.py`)

A FastAPI app that resolves `models:/<dataset>-classifier-prod/Production`
from the registry at startup, where `<dataset>` comes from `DATASET_NAME`.
Run two instances on different ports (e.g. 8000 for wine, 8001 for iris) to
serve both models side by side from the same codebase.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Scale input, run inference, return class + probabilities |
| `GET /health` | Liveness/readiness, includes which dataset is loaded |
| `GET /metrics` | Prometheus-format counters/histograms |
| `POST /reload` | Re-resolve the Production model without restarting the pod |

### 4. Containerization (`Dockerfile`)

A single multi-stage Dockerfile with a shared `base` layer and two leaf
targets (`train`, `serve`). `DATASET_NAME` is passed at container runtime,
not baked into the image, so the same image serves either dataset.

### 5. Orchestration (`k8s/`)

| Manifest | Purpose |
|---|---|
| `00-namespace.yaml` | Isolates all resources under `mlops` |
| `01-mlflow.yaml` | Tracking server + registry backend, backed by a PVC |
| `02-training-job.yaml` | One-shot `Job`; set `DATASET_NAME` env var to pick which model to train |
| `03-serving-deployment.yaml` | `Deployment` (2-8 replicas via HPA), `Service`, pod anti-affinity |
| `04-ingress.yaml` | External routing to `/api` (serving) and `/mlflow` (tracking UI) |
| `05-retrain-cronjob.yaml` | Weekly scheduled retraining |

### 6. CI/CD (`.github/workflows/ci-cd.yml`)

`test → build-and-push → train-and-register → deploy`, each gated on the
previous stage succeeding.

### 7. Observability (`monitoring/prometheus.yml`)

Scrapes the serving API's `/metrics` endpoint.

## Key design decisions & trade-offs

- **Dataset selection via env var, not config branching in code**: keeps
  `preprocess.py`/`train.py`/`serve.py` identical regardless of which model
  is running — the only per-dataset difference is a lookup in
  `DATASET_LOADERS` and a name-templated path/registry key.
- **Separate registered models rather than one multi-output model**: wine
  and iris are unrelated classification problems; registering them as two
  independent MLflow models (rather than forcing a shared schema) keeps the
  promotion gate meaningful for each — a regression on one dataset can never
  accidentally block or trigger a promotion on the other.
- **MLflow over a fully custom registry**: gives experiment tracking, model
  registry, and a UI for free, at the cost of running one more service.
- **Gaussian-jitter augmentation over SMOTE**: chosen for domain-generality;
  swap freely via the same `augment(X, y) -> X, y` interface.
