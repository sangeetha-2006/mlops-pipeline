.PHONY: install test lint preprocess train serve preprocess-iris train-iris serve-iris web up down clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 src --max-line-length=110 --extend-ignore=E203

# ---------------- Wine model (default) ----------------
preprocess:
	cd src && PROCESSED_DATA_DIR=../data/processed DATASET_NAME=wine python preprocess.py

train:
	cd src && PROCESSED_DATA_DIR=../data/processed DATASET_NAME=wine MLFLOW_EXPERIMENT_NAME=wine-classifier REGISTERED_MODEL_NAME=wine-classifier-prod python train.py

serve:
	cd src && DATASET_NAME=wine REGISTERED_MODEL_NAME=wine-classifier-prod SCALER_PATH=../data/processed/wine/scaler.joblib N_FEATURES=13 uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

# ---------------- Iris model (new) ----------------
preprocess-iris:
	cd src && PROCESSED_DATA_DIR=../data/processed DATASET_NAME=iris python preprocess.py

train-iris:
	cd src && PROCESSED_DATA_DIR=../data/processed DATASET_NAME=iris MLFLOW_EXPERIMENT_NAME=iris-classifier REGISTERED_MODEL_NAME=iris-classifier-prod python train.py

serve-iris:
	cd src && DATASET_NAME=iris REGISTERED_MODEL_NAME=iris-classifier-prod SCALER_PATH=../data/processed/iris/scaler.joblib N_FEATURES=4 uvicorn serve:app --host 0.0.0.0 --port 8002 --reload

# ---------------- Web application ----------------
# Browser interface for the whole process: http://localhost:8080
web:
	cd src && uvicorn webapp:app --host 0.0.0.0 --port 8080

# Full local stack: MLflow + both train jobs + both serving APIs + Prometheus
up:
	docker compose up --build

down:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/processed/* .pytest_cache
