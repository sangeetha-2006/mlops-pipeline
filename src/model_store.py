"""
Helpers for loading a registered model together with the scaler it needs.

The training stage stores the fitted scaler inside the same MLflow run as the
model (artifact path "scaler/scaler.joblib"), so a model and the scaling it
expects always travel together. This matters for serving on a cluster, where
the serving pod has no access to the files written by the training pod.

For models trained before this change, a local scaler file can still be used
as a fallback. If neither source exists, loading fails loudly: serving a model
without its scaler would silently return wrong predictions.
"""
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def production_version(client: MlflowClient, name: str, stage: str = "Production"):
    """Return the latest model version in the given stage, or None.

    A model that has never been registered counts as "no version", so a fresh
    installation can be queried before the first training run.
    """
    try:
        versions = client.get_latest_versions(name, stages=[stage])
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    return versions[0] if versions else None


def load_scaler(run_id: str, fallback_path: str = None):
    """Load the scaler stored in the run's artifacts, else from a local file."""
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="scaler/scaler.joblib"
        )
        return joblib.load(local)
    except Exception:
        if fallback_path and os.path.exists(fallback_path):
            return joblib.load(fallback_path)
    raise FileNotFoundError(
        "No scaler found for this model. It was trained before scalers were stored "
        "with the model run. Train the model again so the scaler is saved with it."
    )


def load_bundle(name: str, stage: str = "Production", fallback_scaler_path: str = None,
                client: MlflowClient = None):
    """Load the model in `stage` plus its scaler. Raises LookupError if none exists."""
    client = client or MlflowClient()
    mv = production_version(client, name, stage)
    if mv is None:
        raise LookupError(f"No model named '{name}' is in the {stage} stage yet.")
    model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
    scaler = load_scaler(mv.run_id, fallback_scaler_path)
    return {
        "model": model,
        "scaler": scaler,
        "version": int(mv.version),
        "run_id": mv.run_id,
        "stage": stage,
    }
