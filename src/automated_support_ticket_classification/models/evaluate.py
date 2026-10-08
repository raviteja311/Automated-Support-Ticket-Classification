import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from automated_support_ticket_classification.config import PROJECT_ROOT, load_config
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"


def confusion_frame(y_true, y_pred, labels) -> pd.DataFrame:
    """Confusion matrix with true queues as rows and predicted queues as columns."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )


def errors_frame(test_df: pd.DataFrame, preds, confidence) -> pd.DataFrame:
    """Every misclassified test row, least confident first.

    The original 77-way intent is kept when the corpus has one, so errors can
    be grouped by intent: if a few borderline intents cause most of them, the
    problem is the taxonomy, not the model.
    """
    out = pd.DataFrame(
        {
            "text": test_df["text"].to_numpy(),
            "true_queue": test_df["label"].to_numpy(),
            "predicted_queue": preds,
            "confidence": confidence.round(4),
        }
    )
    if "intent" in test_df.columns:
        out["intent"] = test_df["intent"].to_numpy()
    out = out[out["true_queue"] != out["predicted_queue"]]
    return out.sort_values("confidence").reset_index(drop=True)


def plot_confusion(cm: pd.DataFrame, path: Path) -> None:
    # Imported here: matplotlib is a dev dependency and the serving image
    # never runs this stage.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    threshold = cm.to_numpy().max() / 2
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            value = cm.iat[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    ax.set_title("Confusion matrix, test set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_frame(test_df["label"], preds, list(model.classes_))
    cm.to_csv(REPORTS_DIR / "confusion_matrix.csv")
    plot_confusion(cm, REPORTS_DIR / "confusion_matrix.png")

    confidence = model.predict_proba(test_df["text"]).max(axis=1)
    errors = errors_frame(test_df, preds, confidence)
    errors.to_csv(REPORTS_DIR / "errors.csv", index=False)
    logger.info("%d misclassified test rows written to %s", len(errors), REPORTS_DIR)


if __name__ == "__main__":
    main()
