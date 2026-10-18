"""
Unit tests for the MLOps pipeline. Run with: pytest tests/
These are also what the CI workflow runs before building/pushing an image.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import load_raw_data, clean_data, split_and_scale, DATASET_LOADERS  # noqa: E402
from train import augment, get_candidate_models, evaluate  # noqa: E402


@pytest.mark.parametrize("dataset_name", ["wine", "iris"])
def test_load_raw_data_shape(monkeypatch, dataset_name):
    monkeypatch.setenv("DATASET_NAME", dataset_name)
    import importlib
    import preprocess
    importlib.reload(preprocess)
    df = preprocess.load_raw_data()
    assert "target" in df.columns
    assert len(df) > 0


def test_clean_data_drops_duplicates():
    df = load_raw_data()
    df_with_dupe = df._append(df.iloc[[0]], ignore_index=True) if hasattr(df, "_append") else df.append(df.iloc[[0]], ignore_index=True)
    cleaned = clean_data(df_with_dupe)
    assert len(cleaned) == len(df.drop_duplicates())


def test_split_and_scale_shapes():
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(feature_names) == X_train.shape[1]


def test_augment_increases_dataset_size():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 3, size=10)
    X_aug, y_aug = augment(X, y, factor=2, noise_std=0.01)
    assert X_aug.shape[0] == 30
    assert y_aug.shape[0] == 30


def test_augment_factor_zero_returns_original():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 3, size=10)
    X_aug, y_aug = augment(X, y, factor=0, noise_std=0.01)
    assert X_aug.shape[0] == 10


def test_candidate_models_train_and_evaluate():
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test, scaler, _ = split_and_scale(df)
    models = get_candidate_models()
    assert len(models) == 3
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "f1_weighted" in metrics


def test_dataset_loaders_registered():
    assert "wine" in DATASET_LOADERS
    assert "iris" in DATASET_LOADERS
