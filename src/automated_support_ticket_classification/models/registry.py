"""Model registry: registering versions and gating promotion on a metric.

Training always registers a new version. Promotion is a separate, deliberate
act: a version only becomes `production` if it beats the one already there.
Without that gate, "the model in production" means "whatever the last
`dvc repro` happened to produce", which is not a decision anyone made.

Uses MLflow aliases rather than the deprecated Staging/Production stages.
An alias is a movable pointer to a version, so promotion and rollback are the
same one-line operation in opposite directions.
"""

from __future__ import annotations

from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "ticket-classifier"
PRODUCTION_ALIAS = "production"

# The metric promotion is judged on. Macro F1 rather than accuracy, because
# accuracy can rise while a minority class collapses (see docs/experiments.md).
GATE_METRIC = "f1_macro"


def _client():
    from mlflow import MlflowClient

    return MlflowClient()


def register_version(
    run_id: str, metrics: dict[str, float], data_source: str = "unknown"
) -> str | None:
    """Register the model logged under `run_id` and tag it with its metrics.

    Returns the new version number, or None if MLflow is unavailable.
    Metrics are copied onto the version as tags so promotion can compare
    candidates without re-reading the runs that produced them.
    """
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed; skipping registration")
        return None

    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client = _client()
    for key, value in metrics.items():
        client.set_model_version_tag(MODEL_NAME, result.version, key, f"{value:.6f}")
    # Which corpus produced this score. Without it the gate would compare
    # numbers from different test sets, which is meaningless.
    client.set_model_version_tag(MODEL_NAME, result.version, "data_source", data_source)

    logger.info("Registered %s version %s", MODEL_NAME, result.version)
    return result.version


def production_metric(metric: str = GATE_METRIC) -> tuple[float | None, str | None]:
    """Return (gate metric, data source) of the current production version.

    Both are None when nothing is in production. The data source comes back
    with the metric because a score is only meaningful alongside the test set
    that produced it.
    """
    try:
        version = _client().get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except Exception:
        # No registered model, or no production alias yet. Both mean the same
        # thing to a caller: there is nothing to beat.
        return None, None

    raw = version.tags.get(metric)
    return (float(raw) if raw is not None else None), version.tags.get("data_source")


def promote(version: str) -> None:
    """Point the production alias at `version`."""
    _client().set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
    logger.info("Promoted %s version %s to '%s'", MODEL_NAME, version, PRODUCTION_ALIAS)


def promote_if_better(
    version: str,
    candidate_metric: float,
    metric: str = GATE_METRIC,
    data_source: str = "unknown",
) -> bool:
    """Promote `version` only if it beats a comparable incumbent.

    A tie does not promote: shipping an equal model costs a deploy and gains
    nothing, and the incumbent is already known to work.

    Comparable means trained on the same corpus. Scores from different test
    sets are not on the same scale, so when the corpus changes the incumbent
    is not evidence of anything and the candidate takes over.
    """
    incumbent, incumbent_source = production_metric(metric)

    if incumbent is not None and incumbent_source != data_source:
        logger.info(
            "Production was trained on %r and the candidate on %r; the scores are "
            "not comparable, so promoting version %s on the new corpus",
            incumbent_source,
            data_source,
            version,
        )
        promote(version)
        return True

    if incumbent is None:
        logger.info("No production model yet; promoting version %s by default", version)
        promote(version)
        return True

    if candidate_metric > incumbent:
        logger.info(
            "Candidate %s=%.4f beats production %.4f; promoting version %s",
            metric,
            candidate_metric,
            incumbent,
            version,
        )
        promote(version)
        return True

    logger.info(
        "Candidate %s=%.4f does not beat production %.4f; leaving production unchanged",
        metric,
        candidate_metric,
        incumbent,
    )
    return False
