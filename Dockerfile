# syntax=docker/dockerfile:1

# ---- Base: shared dependency layer ----
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Absolute path so preprocess/train/serve always agree on data location,
# regardless of which directory each stage's ENTRYPOINT runs from.
ENV PROCESSED_DATA_DIR=/app/data/processed

# ---- Training image: runs preprocessing + training, then exits ----
FROM base AS train
WORKDIR /app/src
ENTRYPOINT ["sh", "-c", "python preprocess.py && python train.py"]

# ---- Serving image: long-running API process ----
FROM base AS serve
WORKDIR /app/src
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Web application image: browser interface for the whole pipeline ----
FROM base AS web
WORKDIR /app/src
EXPOSE 8080
ENTRYPOINT ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8080"]
