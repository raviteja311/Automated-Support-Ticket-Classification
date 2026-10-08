"""Is the 5-way task just easier than banking77's 77-way one?

Three numbers on the same official test set:

  direct_5way    the production model, trained on the five queues
  77way          the same pipeline trained on the 77 original intents
  77way_mapped   the 77-way model's predictions folded through INTENT_MAP
                 and scored against the five queues

If 77way_mapped beats direct_5way, the fine-grained labels carry signal the
5-way model throws away, and the better design is to predict intents and map
afterwards. If it does not, the simpler design is backed by data.

YOUR CODE: granularity_scores is a stub. tests/test_granularity.py says what
it must return; remove the xfail marker there once it is implemented.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from automated_support_ticket_classification.config import PROJECT_ROOT, load_config
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

GRANULARITY_PATH = PROJECT_ROOT / "metrics" / "granularity.json"


def granularity_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg) -> dict:
    """Return {"77way": {...}, "77way_mapped": {...}}.

    Each value is {"accuracy": float, "f1_macro": float}, rounded to 4 places.

    Steps:
      1. build_pipeline(cfg) from models/train.py, fit on train_df["intent"].
      2. Score its predictions against test_df["intent"]: the 77way row.
      3. Map those predictions to queues with banking77.INTENT_MAP and score
         them against test_df["label"]: the 77way_mapped row.

    Use the same pipeline settings as production, so the only difference
    from direct_5way is the label granularity.
    """
    raise NotImplementedError("granularity_scores: write this in models/granularity.py")


def main() -> None:
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    if "intent" not in train_df.columns:
        raise ValueError("No intent column: the granularity stage needs data.source: banking77")

    production = joblib.load(cfg.model.model_path)
    preds = production.predict(test_df["text"])
    report = {
        "split": cfg.data.split,
        "test_rows": len(test_df),
        # The shipped artifact, so this row matches metrics/metrics.json.
        "direct_5way": {
            "accuracy": round(float(accuracy_score(test_df["label"], preds)), 4),
            "f1_macro": round(float(f1_score(test_df["label"], preds, average="macro")), 4),
        },
        **granularity_scores(train_df, test_df, cfg),
    }

    GRANULARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRANULARITY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Granularity comparison written to %s: %s", GRANULARITY_PATH, report)


if __name__ == "__main__":
    main()
