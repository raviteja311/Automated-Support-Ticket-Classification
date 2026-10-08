"""Is the 5-way task just easier than banking77's 77-way one?

Three numbers on the same official test set:

  direct_5way    the production model, trained on the five queues
  77way          the same pipeline trained on the 77 original intents
  77way_mapped   the 77-way model's predictions folded through INTENT_MAP
                 and scored against the five queues

If 77way_mapped beats direct_5way, the fine-grained labels carry signal the
5-way model throws away, and the better design is to predict intents and map
afterwards. If it does not, the simpler design is backed by data.

Every row carries a bootstrap 95% CI, and mapped and direct are compared by a
McNemar test on the same test rows, so a small gap can be told from noise.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from automated_support_ticket_classification.config import PROJECT_ROOT, load_config
from automated_support_ticket_classification.data.banking77 import INTENT_MAP
from automated_support_ticket_classification.logger import get_logger
from automated_support_ticket_classification.models.stats import bootstrap_f1_ci, mcnemar_p
from automated_support_ticket_classification.models.train import build_pipeline

logger = get_logger(__name__)

GRANULARITY_PATH = PROJECT_ROOT / "metrics" / "granularity.json"


def _score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def predict_intents(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg) -> np.ndarray:
    """Fit the production pipeline on the 77 intents and predict the test intents.

    Same vectoriser, classifier and settings as production (build_pipeline),
    so the only difference from direct_5way is the label granularity.
    """
    pipe = build_pipeline(cfg).fit(train_df["text"], train_df["intent"])
    return pipe.predict(test_df["text"])


def to_queues(intents) -> np.ndarray:
    """Fold predicted intents into queues through INTENT_MAP."""
    return pd.Series(intents).map(INTENT_MAP).to_numpy()


def granularity_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg) -> dict:
    """Return {"77way": {...}, "77way_mapped": {...}}.

    Each value is {"accuracy": float, "f1_macro": float}, rounded to 4 places.
    77way scores the intent predictions against the true intents; 77way_mapped
    folds the same predictions into queues and scores them against the true
    queues, so it is directly comparable with direct_5way.
    """
    intents = predict_intents(train_df, test_df, cfg)
    return {
        "77way": _score(test_df["intent"], intents),
        "77way_mapped": _score(test_df["label"], to_queues(intents)),
    }


def main() -> None:
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    if "intent" not in train_df.columns:
        raise ValueError("No intent column: the granularity stage needs data.source: banking77")

    seed = cfg.data.random_state
    y_queue = test_df["label"].to_numpy()
    # The shipped artifact, so this row matches metrics/metrics.json.
    direct = joblib.load(cfg.model.model_path).predict(test_df["text"])
    intents = predict_intents(train_df, test_df, cfg)
    mapped = to_queues(intents)

    rows = {
        "direct_5way": (y_queue, direct),
        "77way": (test_df["intent"].to_numpy(), intents),
        "77way_mapped": (y_queue, mapped),
    }
    report = {"split": cfg.data.split, "test_rows": len(test_df)}
    for name, (y_true, y_pred) in rows.items():
        ci_low, ci_high = bootstrap_f1_ci(y_true, y_pred, seed=seed)
        report[name] = {**_score(y_true, y_pred), "ci_low": ci_low, "ci_high": ci_high}

    # The design question: does predicting intents and mapping beat
    # predicting queues directly? Same rows, so a paired test.
    report["mapped_minus_direct_f1_macro"] = round(
        report["77way_mapped"]["f1_macro"] - report["direct_5way"]["f1_macro"], 4
    )
    report["mapped_vs_direct_mcnemar_p"] = mcnemar_p(y_queue, mapped, direct)

    GRANULARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRANULARITY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Granularity comparison written to %s: %s", GRANULARITY_PATH, report)


if __name__ == "__main__":
    main()
