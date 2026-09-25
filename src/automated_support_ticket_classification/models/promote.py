"""Promote the latest registered model to production, if it earns it.

Separate from training on purpose. Training produces a candidate; this decides
whether the candidate ships. Run it by hand, or wire it into CI after the
metrics diff has been reviewed.

    python -m automated_support_ticket_classification.models.promote
"""

import json
import sys
from pathlib import Path

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger
from automated_support_ticket_classification.models.registry import (
    GATE_METRIC,
    MODEL_NAME,
    promote_if_better,
)

logger = get_logger(__name__)


def latest_version() -> str | None:
    from mlflow import MlflowClient

    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))


def main() -> None:
    cfg = load_config()
    metrics_path = Path(cfg.evaluate.metrics_path)
    if not metrics_path.exists():
        logger.error("No metrics at %s. Run the evaluate stage first.", metrics_path)
        sys.exit(1)

    with open(metrics_path, encoding="utf-8") as f:
        summary = json.load(f)["summary"]
    candidate = float(summary[GATE_METRIC])

    version = latest_version()
    if version is None:
        logger.error("Nothing registered as '%s'. Run training with MLflow enabled.", MODEL_NAME)
        sys.exit(1)

    promoted = promote_if_better(version, candidate, data_source=cfg.data.source)

    # Exit non-zero when nothing shipped, so a CI step can branch on it without
    # parsing logs. Not a failure: "the incumbent is still better" is a result.
    sys.exit(0 if promoted else 2)


if __name__ == "__main__":
    main()
