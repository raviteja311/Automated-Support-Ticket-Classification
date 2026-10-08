"""Does a model trained on banking77 transfer to Indian banking customers?

banking77 comes from a European digital bank: top-ups, Apple Pay, exchange
rates. Indian customers write about UPI, IMPS, NEFT, KYC and Aadhaar, often
in Hinglish. data/india_eval/india_messages.csv is a hand-written set of such
messages, labelled with the same five queues, and this stage scores the
production model on it, overall and by language style.

A drop against the banking77 test score is the expected finding, not a
failure: it measures how far the benchmark model is from this market.

The CSV has four columns:

  text            the message, as a customer would type it
  label           one of the five queues
  language_style  english | hinglish | short_or_misspelled
  author          who wrote it, so results can be checked for one person's style

Until messages are added, the stage writes a "no_messages_yet" status rather
than failing, so `dvc repro` keeps working.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from automated_support_ticket_classification.config import PROJECT_ROOT, load_config
from automated_support_ticket_classification.data.banking77 import CATEGORIES
from automated_support_ticket_classification.data.preprocess import clean_text
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

INDIA_PATH = PROJECT_ROOT / "data" / "india_eval" / "india_messages.csv"
OUT_PATH = PROJECT_ROOT / "metrics" / "india.json"
ERRORS_PATH = PROJECT_ROOT / "reports" / "india_errors.csv"
COLUMNS = ["text", "label", "language_style", "author"]
STYLES = ("english", "hinglish", "short_or_misspelled")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on a malformed row; a typo in a label would silently score as an error."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"india_messages.csv is missing columns {missing}")
    df = df.copy()
    for column in ("label", "language_style"):
        df[column] = df[column].astype(str).str.strip().str.lower()
    problems = []
    for i, row in df.iterrows():
        line = i + 2  # header is line 1
        if not str(row["text"]).strip():
            problems.append(f"line {line}: empty text")
        if row["label"] not in CATEGORIES:
            problems.append(f"line {line}: unknown label {row['label']!r}")
        if row["language_style"] not in STYLES:
            problems.append(f"line {line}: unknown language_style {row['language_style']!r}")
    if problems:
        raise ValueError("india_messages.csv has problems:\n  " + "\n  ".join(problems))
    return df


def score(y_true, y_pred) -> dict:
    return {
        "rows": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def evaluate(model, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Scores overall and per language style, plus the misclassified rows."""
    cleaned = df["text"].astype(str).map(clean_text)
    proba = model.predict_proba(cleaned)
    preds = model.classes_[proba.argmax(axis=1)]
    report = {"overall": score(df["label"], preds), "by_style": {}}
    for style in STYLES:
        mask = (df["language_style"] == style).to_numpy()
        if mask.any():
            report["by_style"][style] = score(df["label"][mask], preds[mask])
    errors = df.assign(predicted=preds, confidence=proba.max(axis=1).round(4))
    errors = errors[errors["label"] != errors["predicted"]].sort_values("confidence")
    return report, errors


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(INDIA_PATH, dtype=str).fillna("")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        report = {
            "status": "no_messages_yet",
            "path": INDIA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        }
        logger.warning("No messages in %s yet; nothing to evaluate", INDIA_PATH)
    else:
        df = validate(df)
        model = joblib.load(cfg.model.model_path)
        report, errors = evaluate(model, df)
        report["status"] = "ok"
        # The in-distribution score this stage is compared with.
        banking77 = json.loads(Path(cfg.evaluate.metrics_path).read_text(encoding="utf-8"))
        report["banking77_test_f1_macro"] = round(banking77["summary"]["f1_macro"], 4)
        report["f1_macro_drop"] = round(
            report["banking77_test_f1_macro"] - report["overall"]["f1_macro"], 4
        )
        ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(ERRORS_PATH, index=False)
        logger.info("India set: %s", report)

    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
