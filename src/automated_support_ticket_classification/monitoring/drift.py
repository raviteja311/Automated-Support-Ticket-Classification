"""Drift monitoring: is the model still facing the world it was trained on?

Prometheus answers whether the *service* is healthy: rate, errors, duration.
It cannot tell you that the service is cheerfully returning 200s while the
traffic has changed underneath it. That is what this measures.

Two things are compared against the training set:

  prediction drift  the mix of labels the model is emitting
  text drift        the shape of the incoming text itself

Both matter, and they fail differently. Prediction drift catches the model
suddenly routing everything to one queue. Text drift catches input changing
before the predictions visibly do, which is the earlier and more useful signal.

Reference is the training split, the data the model actually learned from.
Current is whatever the API has served, read from the prediction log.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

# Share of features flagged as drifted, above which the dataset counts as
# drifted overall. Evidently's own default; kept explicit because a threshold
# buried in a library is a threshold nobody reviews.
DRIFT_SHARE_THRESHOLD = 0.5

# Below this many rows of live traffic, no drift verdict is issued at all.
#
# Drift tests compare distributions, and a distribution estimated from a
# handful of requests is mostly noise. Five predictions spread evenly across
# five classes look wildly "drifted" against a training set that is 38% billing
# and 8% shipping, purely because five is too few to estimate a share from.
# Observed directly: a five-row sample reported label drift of 0.1764 against a
# 0.10 threshold, which was an artifact and nothing else.
#
# 200 gives roughly 40 rows per class at the natural balance, which is enough
# for the per-class shares to mean something. An alert that fires on the first
# few requests after a deploy is an alert people learn to ignore.
MIN_SAMPLE_ROWS = 200


def load_predictions(path: Path) -> pd.DataFrame | None:
    """Read the API's prediction log, or None when nothing has been served."""
    if not path.exists():
        return None

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Return an Evidently drift report comparing current traffic to training."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(
        text_columns=["text"],
        categorical_columns=["label"],
    )
    ref = Dataset.from_pandas(reference[["text", "label"]], data_definition=definition)
    cur = Dataset.from_pandas(current[["text", "label"]], data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    return report.run(reference_data=ref, current_data=cur)


def summarise(result, current_rows: int, min_rows: int = MIN_SAMPLE_ROWS) -> dict:
    """Reduce the report to the few numbers worth alerting on.

    The HTML report is for a human investigating. This is for a machine
    deciding whether a human needs to look at the HTML.
    """
    payload = result.dict()
    sufficient = current_rows >= min_rows
    summary: dict = {"columns": {}}

    for metric in payload.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        value = metric.get("value")

        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            summary["columns_drifted"] = int(value.get("count", 0))
            summary["drift_share"] = round(float(value.get("share", 0.0)), 4)

        elif name.startswith("ValueDrift"):
            # e.g. ValueDrift(column=text,method=...,threshold=0.55)
            column = re.search(r"column=([^,)]+)", name)
            threshold = re.search(r"threshold=([0-9.]+)", name)
            method = re.search(r"method=([^,)]+)", name)
            if column and isinstance(value, int | float):
                score = round(float(value), 6)
                limit = float(threshold.group(1)) if threshold else None
                summary["columns"][column.group(1)] = {
                    "score": score,
                    "threshold": limit,
                    "method": method.group(1) if method else None,
                    # Each column carries its own verdict. A single dataset-level
                    # boolean hides which signal moved, and they fail differently:
                    # label drift means the model changed its mind, text drift
                    # means the input changed first.
                    # No verdict at all on a thin sample. "We do not know"
                    # and "no drift" must not look the same.
                    "drifted": bool(limit is not None and score > limit) if sufficient else None,
                }

    summary["columns_checked"] = len(summary["columns"])
    summary["current_rows"] = current_rows
    summary["min_sample_rows"] = min_rows
    summary["sufficient_sample"] = sufficient

    # `status` is what a monitor should key on. Alert only when the sample was
    # large enough to have an opinion AND that opinion is "drifted". Keying on
    # dataset_drifted alone would read an unknown as a clean bill of health.
    if not sufficient:
        summary["status"] = "insufficient_data"
        summary["dataset_drifted"] = None
        summary["note"] = (
            f"{current_rows} rows is below the {min_rows}-row minimum; "
            "scores are reported but no drift verdict is issued."
        )
    else:
        summary["status"] = "ok"
        summary["dataset_drifted"] = bool(summary.get("drift_share", 0.0) > DRIFT_SHARE_THRESHOLD)
    return summary


def main() -> None:
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    reference = pd.read_csv(processed / "train.csv")

    log_path = Path(cfg.data.processed_dir).parent / "predictions.jsonl"
    current = load_predictions(log_path)

    if current is None:
        # No traffic yet. Fall back to the held-out split so the mechanism is
        # still exercised and the report is still produced. Says so loudly,
        # because "no drift" and "no data" must never look the same.
        logger.warning(
            "No predictions at %s; comparing the test split against train instead. "
            "This exercises the report but is not a measurement of real traffic.",
            log_path,
        )
        current = pd.read_csv(processed / "test.csv")

    result = build_report(reference, current)

    out_dir = Path(cfg.evaluate.metrics_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "drift_report.html"
    result.save_html(str(html_path))

    summary = summarise(result, current_rows=len(current))
    summary["reference_rows"] = len(reference)

    if summary["status"] == "insufficient_data":
        logger.warning("%s", summary["note"])

    json_path = out_dir / "drift.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Drift summary: %s", summary)
    logger.info("Report written to %s", html_path)


if __name__ == "__main__":
    main()
