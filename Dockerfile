# ---------- Stage 1: build dependencies ----------
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-serve.txt .
RUN pip install --prefix=/install -r requirements-serve.txt

# ---------- Stage 2: lean runtime ----------
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src DISABLE_MLFLOW=1
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY params.yaml ./

# Bake the model into the image: it trains in seconds and keeps the
# container self-contained, with no external storage to configure.
# config.py resolves paths from its own location, so params.yaml at
# /app is found regardless of the working directory.
RUN python -m automated_support_ticket_classification.data.generate \
    && python -m automated_support_ticket_classification.data.preprocess \
    && python -m automated_support_ticket_classification.models.train

EXPOSE 8000

# Render and similar platforms inject $PORT; fall back to 8000 locally.
# Bind 0.0.0.0, not 127.0.0.1, or nothing outside the container can reach it.
CMD ["sh", "-c", "uvicorn automated_support_ticket_classification.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
