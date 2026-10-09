"""LogReg or calibrated LinearSVC: decide by a rule written before the run.

LinearSVC scores higher macro F1 than the shipped LogReg (baselines.json), but
LogReg was kept because LinearSVC has no probabilities. Calibration removes
that reason, so the choice has to be made on evidence:

  f1_macro      with a bootstrap 95% CI (models/stats.py)
  mcnemar_p     are the two models' errors really different?
  ece           expected calibration error: does "80% confident" mean right
                80% of the time? The needs-review threshold depends on it.
  latency       predict_proba on one message, model only

The decision rule lives in docs/model-decision.md and must be written and
committed before this runs; the script refuses while the file still contains
TODO, and records the rule's SHA-256 in the output so the result is tied to
the rule exactly as it was written. Deciding the rule after seeing the
numbers would make it a description, not a test.

This is run once, by hand, not as a DVC stage:
    python -m automated_support_ticket_classification.models.compare
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from automated_support_ticket_classification.config import PROJECT_ROOT, load_config
from automated_support_ticket_classification.logger import get_logger
from automated_support_ticket_classification.models.stats import bootstrap_f1_ci, mcnemar_p
from automated_support_ticket_classification.models.train import build_pipeline

logger = get_logger(__name__)

RULE_PATH = PROJECT_ROOT / "docs" / "model-decision.md"
OUT_PATH = PROJECT_ROOT / "metrics" / "model_comparison.json"
PLOT_PATH = PROJECT_ROOT / "reports" / "reliability.png"
COMPARED = ("logreg", "linearsvc_calibrated")
N_BINS = 10


def reliability_table(confidence, correct, n_bins: int = N_BINS) -> pd.DataFrame:
    """Bin predictions by top-label confidence and compare confidence with accuracy.

    Use n_bins equal-width bins over [0, 1]; a confidence of exactly 1.0 goes in
    the last bin. Return one row per non-empty bin with columns:

      bin_low, bin_high   the bin's edges
      count               predictions in the bin
      mean_confidence     their average confidence
      accuracy            the share of them that were correct

    A perfectly calibrated model has accuracy == mean_confidence in every bin.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    # Bin index by multiplication rather than comparing with np.linspace
    # edges: linspace(0, 1, 11)[7] is 0.7000000000000001, which would push a
    # confidence of exactly 0.7 into the bin below. min() puts 1.0 in the
    # last bin instead of an eleventh one.
    bins = np.minimum(np.floor(confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bins == b
        if not in_bin.any():
            continue
        rows.append(
            {
                "bin_low": b / n_bins,
                "bin_high": (b + 1) / n_bins,
                "count": int(in_bin.sum()),
                "mean_confidence": float(confidence[in_bin].mean()),
                "accuracy": float(correct[in_bin].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(confidence, correct, n_bins: int = N_BINS) -> float:
    """ECE: the count-weighted mean of |accuracy - mean_confidence| over the bins.

    Weighting by count means a badly calibrated bin with a handful of
    predictions barely moves ECE, while one holding most of the traffic
    dominates it: it measures miscalibration where predictions actually fall.
    """
    table = reliability_table(confidence, correct, n_bins)
    gaps = (table["accuracy"] - table["mean_confidence"]).abs()
    return round(float((table["count"] * gaps).sum() / table["count"].sum()), 4)


def check_rule(path: Path = RULE_PATH) -> str:
    """Return the rule file's SHA-256, or stop if the rule is not written yet."""
    if not path.exists() or "TODO" in path.read_text(encoding="utf-8"):
        raise SystemExit(
            f"Write and commit the decision rule in {path} before comparing models. "
            "It must not contain TODO."
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency_ms(pipe, texts: list[str], warmup: int = 20) -> dict:
    """predict_proba on one message at a time, model only."""
    for text in texts[:warmup]:
        pipe.predict_proba([text])
    samples = []
    for text in texts:
        start = time.perf_counter()
        pipe.predict_proba([text])
        samples.append((time.perf_counter() - start) * 1000)
    q = statistics.quantiles(samples, n=100, method="inclusive")
    return {"p50_ms": round(statistics.median(samples), 3), "p95_ms": round(q[94], 3)}


def plot_reliability(tables: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    for name, table in tables.items():
        ax.plot(table["mean_confidence"], table["accuracy"], marker="o", label=name)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title("Reliability diagram, test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    rule_sha256 = check_rule()
    cfg = load_config()
    processed = Path(cfg.data.processed_dir)
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    y_test = test_df["label"].to_numpy()
    seed = cfg.data.random_state
    texts = test_df["text"].astype(str).tolist()

    results, preds, tables = {}, {}, {}
    for model_type in COMPARED:
        pipe = build_pipeline(cfg, model_type).fit(train_df["text"], train_df["label"])
        proba = pipe.predict_proba(test_df["text"])
        preds[model_type] = pipe.classes_[proba.argmax(axis=1)]
        confidence = proba.max(axis=1)
        correct = preds[model_type] == y_test
        ci_low, ci_high = bootstrap_f1_ci(y_test, preds[model_type], seed=seed)
        tables[model_type] = reliability_table(confidence, correct)
        results[model_type] = {
            "f1_macro": round(float(f1_score(y_test, preds[model_type], average="macro")), 4),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ece": expected_calibration_error(confidence, correct),
            "latency": latency_ms(pipe, texts[:500]),
        }
        logger.info("%s: %s", model_type, results[model_type])

    report = {
        "split": cfg.data.split,
        "test_rows": len(test_df),
        "rule_file": str(RULE_PATH.relative_to(PROJECT_ROOT)),
        "rule_sha256": rule_sha256,
        "models": results,
        "mcnemar_p": mcnemar_p(y_test, preds["logreg"], preds["linearsvc_calibrated"]),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_reliability(tables, PLOT_PATH)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
