"""Measure how far other people agree with the 77-intent to 5-queue mapping.

INTENT_MAP is one person's judgement. This turns it into a measured one:

  sheet        write a blind annotation sheet: every intent with three example
               messages and an empty queue column. INTENT_MAP is not shown, so
               annotators are not anchored to it.
  agree        read the completed sheets from docs/mapping/annotations/, score
               agreement with INTENT_MAP (Cohen's kappa per pair, Fleiss' kappa
               for everyone), list the intents people disputed, and write the
               majority-vote mapping.
  sensitivity  retrain on the official split with the majority-vote mapping
               and report how far macro F1 moves.

Usage:
    python -m automated_support_ticket_classification.data.mapping_validation sheet
    python -m automated_support_ticket_classification.data.mapping_validation agree
    python -m automated_support_ticket_classification.data.mapping_validation sensitivity
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from automated_support_ticket_classification.config import PROJECT_ROOT, load_config
from automated_support_ticket_classification.data.banking77 import CATEGORIES, INTENT_MAP
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

MAPPING_DIR = PROJECT_ROOT / "docs" / "mapping"
SHEET_PATH = MAPPING_DIR / "annotation_sheet.csv"
ANNOTATIONS_DIR = MAPPING_DIR / "annotations"
MAJORITY_PATH = MAPPING_DIR / "majority_map.csv"
REPORT_PATH = PROJECT_ROOT / "metrics" / "mapping_agreement.json"

# The name INTENT_MAP's author goes by in the agreement report.
AUTHOR = "author"
EXAMPLES_PER_INTENT = 3


def build_sheet(train: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per intent, alphabetical, with example messages and a blank queue."""
    rows = []
    for intent, group in sorted(train.groupby("category"), key=lambda kv: kv[0].lower()):
        examples = group["text"].sample(n=EXAMPLES_PER_INTENT, random_state=seed).tolist()
        rows.append(
            {
                "intent": intent,
                **{f"example_{i + 1}": text for i, text in enumerate(examples)},
                "queue": "",
            }
        )
    return pd.DataFrame(rows)


def load_annotations(directory: Path) -> dict[str, dict[str, str]]:
    """Return {annotator: {intent: queue}} from every CSV in `directory`.

    Fails loudly on a missing intent or an unknown queue name, because a
    half-filled sheet would quietly lower every kappa.
    """
    annotations = {}
    for path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        mapping = dict(zip(df["intent"], df["queue"].str.strip().str.lower(), strict=True))
        missing = sorted(set(INTENT_MAP) - {k for k, v in mapping.items() if v})
        unknown = sorted({v for v in mapping.values() if v and v not in CATEGORIES})
        if missing or unknown:
            raise ValueError(f"{path.name}: unlabelled intents {missing}, unknown queues {unknown}")
        annotations[path.stem] = mapping
    return annotations


def agreement(mappings: dict[str, dict[str, str]]) -> dict:
    """Cohen's kappa for each pair, Fleiss' kappa for all, and disputed intents."""
    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

    intents = sorted(INTENT_MAP)
    names = sorted(mappings)
    labels = {name: [mappings[name][i] for i in intents] for name in names}

    pairwise = {
        f"{a}_vs_{b}": round(float(cohen_kappa_score(labels[a], labels[b])), 4)
        for a, b in combinations(names, 2)
    }
    table = pd.DataFrame(labels, index=intents)
    codes = {queue: i for i, queue in enumerate(CATEGORIES)}
    counts, _ = aggregate_raters(table.apply(lambda col: col.map(codes)).to_numpy())
    disputed = {intent: dict(row) for intent, row in table.iterrows() if row.nunique() > 1}
    return {
        "raters": names,
        "intents": len(intents),
        "cohen_kappa": pairwise,
        "fleiss_kappa": round(float(fleiss_kappa(counts)), 4),
        "unanimous": len(intents) - len(disputed),
        "disputed": disputed,
    }


def majority_map(mappings: dict[str, dict[str, str]]) -> dict[str, str]:
    """Most common queue per intent. A tie keeps INTENT_MAP's choice."""
    result = {}
    for intent, original in INTENT_MAP.items():
        votes = Counter(m[intent] for m in mappings.values()).most_common()
        top = [queue for queue, n in votes if n == votes[0][1]]
        result[intent] = original if original in top or len(top) > 1 else top[0]
    return result


def sensitivity(alternative: dict[str, str], cfg) -> dict:
    """Macro F1 on the official split under INTENT_MAP and under `alternative`.

    Each mapping is scored against its own labels: the question is whether the
    task stays as learnable, not whether one mapping predicts the other.
    """
    from sklearn.metrics import f1_score

    from automated_support_ticket_classification.data.preprocess import clean_text
    from automated_support_ticket_classification.models.train import build_pipeline

    cache = Path(cfg.data.cache_dir)
    train = pd.read_csv(cache / "train.csv")
    test = pd.read_csv(cache / "test.csv")
    x_train = train["text"].astype(str).map(clean_text)
    x_test = test["text"].astype(str).map(clean_text)

    scores = {}
    for name, mapping in (("intent_map", INTENT_MAP), ("majority_map", alternative)):
        pipe = build_pipeline(cfg).fit(x_train, train["category"].map(mapping))
        preds = pipe.predict(x_test)
        y_test = test["category"].map(mapping)
        scores[name] = round(float(f1_score(y_test, preds, average="macro")), 4)
    changed = sorted(i for i in INTENT_MAP if INTENT_MAP[i] != alternative[i])
    return {
        "f1_macro": scores,
        "delta_f1_macro": round(scores["majority_map"] - scores["intent_map"], 4),
        "intents_moved": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("sheet", "agree", "sensitivity"))
    args = parser.parse_args()
    cfg = load_config()

    if args.command == "sheet":
        from automated_support_ticket_classification.data import banking77

        banking77._download(Path(cfg.data.cache_dir))
        train = pd.read_csv(Path(cfg.data.cache_dir) / "train.csv")
        MAPPING_DIR.mkdir(parents=True, exist_ok=True)
        build_sheet(train, seed=cfg.data.random_state).to_csv(SHEET_PATH, index=False)
        logger.info("Blank annotation sheet written to %s", SHEET_PATH)
        return

    annotations = load_annotations(ANNOTATIONS_DIR)
    if not annotations:
        raise SystemExit(f"No completed sheets in {ANNOTATIONS_DIR}; see docs/mapping/README.md")
    mappings = {AUTHOR: INTENT_MAP, **annotations}
    majority = majority_map(mappings)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    if args.command == "agree":
        report["agreement"] = agreement(mappings)
        pd.DataFrame(sorted(majority.items()), columns=["intent", "queue"]).to_csv(
            MAJORITY_PATH, index=False
        )
    else:
        report["sensitivity"] = sensitivity(majority, cfg)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
