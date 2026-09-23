from functools import lru_cache

import joblib
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from automated_support_ticket_classification.api.schemas import TicketRequest, TicketResponse
from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.data.preprocess import clean_text
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Automated Support Ticket Classification API", version="1.0.0")

# Adds request count, latency and error-rate metrics, and serves them at
# /metrics for Prometheus to scrape. The RED signals: Rate, Errors, Duration.
Instrumentator().instrument(app).expose(app)


@lru_cache
def get_model():
    """Load the pipeline once, lazily, so a missing file cannot stop the app starting."""
    cfg = load_config()
    logger.info("Loading model from %s", cfg.model.model_path)
    return joblib.load(cfg.model.model_path)


@app.get("/")
def root() -> dict:
    return {
        "service": "automated-support-ticket-classification",
        "docs": "/docs",
        "health": "/health",
    }


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

    model = get_model()
    proba = model.predict_proba([text])[0]
    # strict=True: classes_ and the probability row must be the same length.
    # If they ever are not, fail loudly rather than silently truncating.
    scores = {cls: float(p) for cls, p in zip(model.classes_, proba, strict=True)}
    best = max(scores, key=scores.get)
    return TicketResponse(label=best, confidence=scores[best], all_scores=scores)
