"""
Training stage of the MLOps pipeline.

- Applies a light data-augmentation step (Gaussian jitter) so the model
  sees a more varied training distribution ("augmented" ML).
- Trains a set of candidate models, logs params/metrics/artifacts to
  MLflow, and registers the best-performing run in the MLflow Model
  Registry, promoting it to "Staging" automatically if it beats the
  current "Production" model on the test set.

Dataset-aware: reads from PROCESSED_DATA_DIR/<DATASET_NAME>/ (written by
preprocess.py) and registers under a model name specific to that dataset,
so "wine" and "iris" models can be trained and promoted independently.
"""
import os
import json
import logging

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv("DATASET_NAME", "wine").lower()
BASE_PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
PROCESSED_DATA_DIR = os.path.join(BASE_PROCESSED_DIR, DATASET_NAME)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", f"{DATASET_NAME}-classifier")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{DATASET_NAME}-classifier-prod")
AUGMENT_FACTOR = int(os.getenv("AUGMENT_FACTOR", 2))   # how many noisy copies per sample
NOISE_STD = float(os.getenv("NOISE_STD", 0.03))


def augment(X: np.ndarray, y: np.ndarray, factor: int, noise_std: float):
    """Simple Gaussian-jitter augmentation for tabular data.

    Creates `factor` noisy copies of the training set. This is a
    stand-in for whatever domain-appropriate augmentation is relevant
    (SMOTE for imbalance, mixup, synthetic records, etc.) — swap the
    body of this function without touching the rest of the pipeline.
    """
    if factor <= 0:
        return X, y
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(factor):
        noise = np.random.normal(0, noise_std, X.shape)
        augmented_X.append(X + noise)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.concatenate([np.asarray(a) for a in augmented_y])


def get_candidate_models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, preds, average="weighted"),
    }


def get_current_production_metric(client: MlflowClient, metric_key="f1_weighted"):
    """Look up the f1 score of whatever is currently in Production, if anything."""
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return run.data.metrics.get(metric_key)
    except Exception:
        return None


def run_training():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    logger.info(
        "Dataset: %s | Experiment: %s | Registry name: %s",
        DATASET_NAME, EXPERIMENT_NAME, REGISTERED_MODEL_NAME,
    )

    data = joblib.load(os.path.join(PROCESSED_DATA_DIR, "dataset.joblib"))
    scaler_path = os.path.join(PROCESSED_DATA_DIR, "scaler.joblib")
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    logger.info("Augmenting training data (factor=%d, noise_std=%.3f)...", AUGMENT_FACTOR, NOISE_STD)
    X_train_aug, y_train_aug = augment(X_train, np.asarray(y_train), AUGMENT_FACTOR, NOISE_STD)
    logger.info("Training set size: %d -> %d after augmentation", len(X_train), len(X_train_aug))

    best_run_id, best_model_name, best_f1 = None, None, -1.0
    candidates = []

    for name, model in get_candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("dataset", DATASET_NAME)
            mlflow.log_param("model_type", name)
            mlflow.log_param("augment_factor", AUGMENT_FACTOR)
            mlflow.log_param("noise_std", NOISE_STD)
            mlflow.log_params(model.get_params())

            model.fit(X_train_aug, y_train_aug)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            # Keep the fitted scaler with the run, so the model and the scaling
            # it expects always travel together (needed when serving on a cluster).
            if os.path.exists(scaler_path):
                mlflow.log_artifact(scaler_path, artifact_path="scaler")

            logger.info("[%s] metrics: %s", name, metrics)
            candidates.append({
                "model": name,
                "run_id": run.info.run_id,
                **{k: round(float(v), 4) for k, v in metrics.items()},
            })

            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                best_run_id = run.info.run_id
                best_model_name = name

    logger.info("Best candidate: %s (f1_weighted=%.4f, run_id=%s)", best_model_name, best_f1, best_run_id)

    # Register the best run's model
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=registered.version, stage="Staging"
    )
    logger.info("Registered %s v%s -> Staging", REGISTERED_MODEL_NAME, registered.version)

    # Gatekeeping: only promote to Production if it beats the incumbent
    current_prod_f1 = get_current_production_metric(client)
    promoted = current_prod_f1 is None or best_f1 > current_prod_f1
    if promoted:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Promoted v%s to Production (new f1=%.4f vs previous=%s)",
            registered.version, best_f1, current_prod_f1,
        )
    else:
        logger.info(
            "Kept existing Production model (f1=%.4f) over new candidate (f1=%.4f)",
            current_prod_f1, best_f1,
        )

    return {
        "dataset": DATASET_NAME,
        "registered_model": REGISTERED_MODEL_NAME,
        "version": int(registered.version),
        "best_model": best_model_name,
        "best_f1": round(float(best_f1), 4),
        "previous_f1": None if current_prod_f1 is None else round(float(current_prod_f1), 4),
        "promoted": bool(promoted),
        "augment_factor": AUGMENT_FACTOR,
        "noise_std": NOISE_STD,
        "train_size_before": int(len(X_train)),
        "train_size_after": int(len(X_train_aug)),
        "candidates": candidates,
    }


if __name__ == "__main__":
    summary = run_training()
    # One machine-readable line at the end, read by the web application.
    print("RESULT_JSON:" + json.dumps(summary))
