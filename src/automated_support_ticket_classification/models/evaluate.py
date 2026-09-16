import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
