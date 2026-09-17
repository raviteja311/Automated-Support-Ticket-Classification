import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    model = joblib.load(cfg.model.model_path)
    test_df = pd.read_csv(Path(cfg.data.processed_dir) / "test.csv")
    preds = model.predict(test_df["text"])
    summary = {
        "accuracy": float(accuracy_score(test_df["label"], preds)),
        "f1_macro": float(f1_score(test_df["label"], preds, average="macro")),
    }
    per_class = classification_report(test_df["label"], preds, output_dict=True)
    out = Path(cfg.evaluate.metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_class": per_class}, f, indent=2)
    logger.info("Metrics written to %s: %s", out, summary)


if __name__ == "__main__":
    main()
