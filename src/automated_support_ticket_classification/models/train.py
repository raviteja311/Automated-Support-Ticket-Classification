import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger
from automated_support_ticket_classification.models.registry import register_version

logger = get_logger(__name__)
USE_MLFLOW = os.getenv("DISABLE_MLFLOW", "0") != "1"
try:
    import mlflow
except ImportError:
    mlflow = None
    USE_MLFLOW = False


MODEL_TYPES = ("logreg", "linearsvc_calibrated")


def make_classifier(cfg, model_type: str | None = None):
    """The classifier for `model_type`, defaulting to the configured model.type.

    LinearSVC has no predict_proba, and the API needs probabilities for its
    confidence score and the needs-review route. CalibratedClassifierCV fits
    the SVC on cross-validation folds and learns a mapping from its decision
    scores to probabilities on the held-out fold.
    """
    model_type = model_type or cfg.model.type
    if model_type == "logreg":
        return LogisticRegression(C=cfg.model.C, max_iter=cfg.model.max_iter)
    if model_type == "linearsvc_calibrated":
        return CalibratedClassifierCV(
            LinearSVC(C=cfg.model.C, random_state=cfg.data.random_state),
            method=cfg.model.calibration,
            cv=5,
        )
    raise ValueError(f"Unknown model.type {model_type!r}; expected one of {MODEL_TYPES}")


def build_pipeline(cfg, model_type: str | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg.model.max_features,
                    ngram_range=(1, cfg.model.ngram_max),
                ),
            ),
            (
                "clf",
                make_classifier(cfg, model_type),
            ),
        ]
    )


def main() -> None:
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    pipeline = build_pipeline(cfg)

    if USE_MLFLOW and mlflow is not None:
        mlflow.set_experiment("ticket-triage")
        mlflow.start_run()
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "max_features": cfg.model.max_features,
                "ngram_max": cfg.model.ngram_max,
                "C": cfg.model.C,
            }
        )

    pipeline.fit(train_df["text"], train_df["label"])
    preds = pipeline.predict(test_df["text"])
    acc = float(accuracy_score(test_df["label"], preds))
    f1 = float(f1_score(test_df["label"], preds, average="macro"))
    logger.info("accuracy=%.4f f1_macro=%.4f", acc, f1)

    model_path = Path(cfg.model.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    if USE_MLFLOW and mlflow is not None:
        metrics = {"accuracy": acc, "f1_macro": f1}
        mlflow.log_metrics(metrics)

        # log_model, not log_artifact: the registry needs a runs:/<id>/model
        # URI, which only the flavour-aware logger produces.
        run_id = mlflow.active_run().info.run_id
        # artifact_path, not name: MLflow 2.x signature. 3.x renamed it.
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.end_run()

        # Register every run. Promotion is a separate, gated decision:
        # see models/promote.py.
        register_version(run_id, metrics, data_source=cfg.data.source)


if __name__ == "__main__":
    main()
