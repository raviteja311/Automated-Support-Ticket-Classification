from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

from automated_support_ticket_classification.api.schemas import TicketRequest, TicketResponse
from automated_support_ticket_classification.api.ui import INDEX_HTML
from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.data.preprocess import clean_text
from automated_support_ticket_classification.logger import get_logger
from automated_support_ticket_classification.monitoring import prediction_log

logger = get_logger(__name__)

app = FastAPI(title="Automated Support Ticket Classification API", version="1.0.0")

# Adds request count, latency and error-rate metrics, and serves them at
# /metrics for Prometheus to scrape. The RED signals: Rate, Errors, Duration.
Instrumentator().instrument(app).expose(app)


class ModelUnavailableError(RuntimeError):
    """The model artifact is missing, so predictions cannot be served."""


@lru_cache
def get_model():
    """Load the pipeline once, lazily, so a missing file cannot stop the app starting.

    A missing artifact raises ModelUnavailableError rather than joblib's
    FileNotFoundError, so /predict can answer 503 instead of a bare 500.
    lru_cache does not cache exceptions, so once the file appears the next
    request loads it without a restart.
    """
    cfg = load_config()
    path = Path(cfg.model.model_path)
    if not path.exists():
        raise ModelUnavailableError(f"Model artifact not found at {path}")
    logger.info("Loading model from %s", path)
    return joblib.load(path)


@app.get("/")
def root() -> dict:
    return {
        "service": "automated-support-ticket-classification",
        "ui": "/ui",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    """A small hand-written page for trying the classifier.

    Excluded from the OpenAPI schema: it is a convenience for humans, not part
    of the API contract that /docs describes.
    """
    return INDEX_HTML


def _log_prediction(text: str, label: str) -> None:
    """Append one prediction to the log the drift monitor reads.

    Best effort on purpose: a monitoring side-effect must never be able to
    fail a request. If the disk is full the user still gets their answer.
    """
    try:
        mon = load_config().monitoring
        # Redacted before it touches disk; see monitoring/prediction_log.py.
        logged = prediction_log.redact(text) if mon.redact else text
        prediction_log.append(
            Path(mon.predictions_path),
            {"text": logged, "label": label},
            max_bytes=mon.max_bytes,
            backup_count=mon.backup_count,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not write to the prediction log", exc_info=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest) -> TicketResponse:
    # Same normalisation as training. Reuse it, never restate it: two
    # definitions of "clean" is how train/serve skew starts.
    text = clean_text(request.text)
    if not text:
        # Reachable for whitespace-only input, which passes min_length=1.
        raise HTTPException(status_code=422, detail="text must not be empty")

    try:
        model = get_model()
    except ModelUnavailableError:
        logger.error("Prediction refused: no model artifact", exc_info=False)
        # 503, not 500: the service is up but cannot serve yet, which is a
        # state, not a bug. Run `dvc repro` (or `dvc pull`) to produce the file.
        raise HTTPException(
            status_code=503,
            detail="Model not available: the model artifact is missing. "
            "Train it with `dvc repro` or fetch it with `dvc pull`.",
        ) from None

    proba = model.predict_proba([text])[0]
    # strict=True: classes_ and the probability row must be the same length.
    # If they ever are not, fail loudly rather than silently truncating.
    scores = {cls: float(p) for cls, p in zip(model.classes_, proba, strict=True)}
    best = max(scores, key=scores.get)

    # Feeds monitoring/drift.py. Without a record of what the model actually
    # saw in production, drift cannot be measured at all.
    _log_prediction(text, best)

    return TicketResponse(label=best, confidence=scores[best], all_scores=scores)
