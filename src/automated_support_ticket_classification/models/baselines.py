"""Put the production model's score in context.

0.92 macro F1 means little on its own. This stage answers two questions on the
exact split and features the production model uses:

  floor        what does a model that has learned nothing score? A majority
               class predictor shows how much of the number is class balance.
  alternatives would a different linear model on the same TF-IDF features do
               better? LinearSVC and ComplementNB are the standard comparisons.

Every score carries a bootstrap 95% confidence interval, and the production
model is compared with the best alternative by a McNemar test on the same test
rows, so a gap can be told apart from noise.

It reports, it does not decide. The production model is only flagged as beaten
when an alternative wins by more than CLEAR_WIN_MARGIN macro F1; swapping it is
a separate, deliberate change to train.py, not a side effect of this stage.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger
from automated_support_ticket_classification.models.stats import bootstrap_f1_ci, mcnemar_p
from automated_support_ticket_classification.models.train import build_pipeline

logger = get_logger(__name__)

PRODUCTION = "tfidf_logreg"

# One percentage point of macro F1. Smaller gaps are within what a different
# random split would move, so they are not a reason to change models.
CLEAR_WIN_MARGIN = 0.01


def candidates(cfg) -> dict:
    """Alternative estimators, each on the production TF-IDF settings."""
    models = {}
    for name, clf in (
        ("majority_class", DummyClassifier(strategy="most_frequent")),
        ("tfidf_linearsvc", LinearSVC(C=cfg.model.C, random_state=cfg.data.random_state)),
        ("tfidf_complementnb", ComplementNB()),
    ):
        pipe = build_pipeline(cfg)
        # Swap only the classifier, so the vectoriser is configured identically.
        pipe.set_params(clf=clf)
        models[name] = pipe
    return models


def score(y_true, y_pred, seed: int) -> dict:
    ci_low, ci_high = bootstrap_f1_ci(y_true, y_pred, seed=seed)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")

    seed = cfg.data.random_state
    results, predictions = {}, {}
    for name, pipe in candidates(cfg).items():
        pipe.fit(train_df["text"], train_df["label"])
        predictions[name] = pipe.predict(test_df["text"])
        results[name] = score(test_df["label"], predictions[name], seed)
        logger.info("%s: %s", name, results[name])

    # The production row is the shipped artifact itself, not a refit, so it
    # matches metrics/metrics.json exactly.
    production = joblib.load(cfg.model.model_path)
    predictions[PRODUCTION] = production.predict(test_df["text"])
    results[PRODUCTION] = score(test_df["label"], predictions[PRODUCTION], seed)
    logger.info("%s (production): %s", PRODUCTION, results[PRODUCTION])

    alternatives = {k: v for k, v in results.items() if k not in (PRODUCTION, "majority_class")}
    best = max(alternatives, key=lambda k: alternatives[k]["f1_macro"])
    margin = round(alternatives[best]["f1_macro"] - results[PRODUCTION]["f1_macro"], 4)

    report = {
        "test_rows": len(test_df),
        "production": PRODUCTION,
        "models": results,
        "best_alternative": best,
        "best_alternative_margin_f1_macro": margin,
        "clear_win_margin": CLEAR_WIN_MARGIN,
        # Same test rows, so a paired test. Below 0.05: the gap is unlikely
        # to be an accident of which messages landed in the test set.
        "mcnemar_p": mcnemar_p(test_df["label"], predictions[PRODUCTION], predictions[best]),
        "production_beaten": bool(margin > CLEAR_WIN_MARGIN),
    }

    out = Path(cfg.evaluate.baselines_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Baselines written to %s", out)
    if report["production_beaten"]:
        logger.warning(
            "%s beats production by %.4f macro F1; consider changing train.py", best, margin
        )


if __name__ == "__main__":
    main()
