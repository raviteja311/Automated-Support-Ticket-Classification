"""Measure /predict latency, end to end and model only.

Two numbers, because they answer different questions:

  api    POST /predict through FastAPI's TestClient, in process. Includes
         validation, clean_text, the model, JSON encoding and the prediction
         log write. Excludes the network and uvicorn's own overhead.
  model  pipeline.predict_proba on one already-cleaned message. The floor:
         what the API could never beat.

Texts are drawn from the held-out split, so lengths are realistic. Prediction
log writes go to a temporary file, so benchmarking never pollutes the log the
drift monitor reads.

Numbers depend heavily on the machine. Treat them as an order of magnitude for
this laptop, not a service-level objective.

Usage:
    python scripts/benchmark_latency.py            # N=500
    python scripts/benchmark_latency.py -n 1000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from automated_support_ticket_classification.api import app as app_module
from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.data.preprocess import clean_text

WARMUP = 20


def percentiles(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    q = statistics.quantiles(ordered, n=100, method="inclusive")
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(q[94], 3),
        "p99_ms": round(q[98], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "n": len(ordered),
    }


def load_texts(n: int) -> list[str]:
    cfg = load_config()
    test_csv = Path(cfg.data.processed_dir) / "test.csv"
    if test_csv.exists():
        texts = pd.read_csv(test_csv)["text"].astype(str).tolist()
    else:
        texts = ["I was charged twice this month", "my card has not arrived yet"]
    return [texts[i % len(texts)] for i in range(n)]


def time_calls(fn, texts: list[str]) -> list[float]:
    for text in texts[:WARMUP]:
        fn(text)
    samples = []
    for text in texts:
        start = time.perf_counter()
        fn(text)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", type=int, default=500, help="timed requests per measurement")
    args = parser.parse_args()
    if args.n < 200:
        sys.exit("Use n >= 200; fewer samples make p95 meaningless.")

    texts = load_texts(args.n)

    with tempfile.TemporaryDirectory() as tmp:
        real_load_config = app_module.load_config

        def load_config_with_scratch_log():
            cfg = real_load_config()
            cfg.monitoring.predictions_path = str(Path(tmp) / "predictions.jsonl")
            return cfg

        app_module.load_config = load_config_with_scratch_log
        try:
            client = TestClient(app_module.app)

            def call_api(text: str) -> None:
                response = client.post("/predict", json={"text": text})
                response.raise_for_status()

            api = percentiles(time_calls(call_api, texts))
        finally:
            app_module.load_config = real_load_config

    model = app_module.get_model()
    cleaned = [clean_text(t) for t in texts]
    model_only = percentiles(time_calls(lambda t: model.predict_proba([t]), cleaned))

    result = {
        "api_predict": api,
        "model_predict_proba": model_only,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
