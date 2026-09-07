from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)


def build_pipeline(cfg) -> Pipeline:
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
                LogisticRegression(C=cfg.model.C, max_iter=cfg.model.max_iter),
            ),
        ]
    )


def main() -> None:
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    pipeline = build_pipeline(cfg)
    pipeline.fit(train_df["text"], train_df["label"])
    preds = pipeline.predict(test_df["text"])
    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds, average="macro")
    logger.info("accuracy=%.4f f1_macro=%.4f", acc, f1)
    model_path = Path(cfg.model.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)


if __name__ == "__main__":
    main()
