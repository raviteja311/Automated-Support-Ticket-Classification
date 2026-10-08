"""Measure how much PII the prediction-log redaction actually catches.

Runs `prediction_log.redact` over the hand-written cases in
tests/fixtures/pii_cases.csv and reports two numbers:

  recall      share of PII values that no longer appear in the output, by kind
  untouched   share of PII-free messages that come out unchanged. Redaction
              that also eats amounts and order numbers costs routing signal.

All values in the fixture are made up. The result is written to
metrics/pii_redaction.json so the README can quote it.

Usage:
    python scripts/pii_recall.py
"""

from __future__ import annotations

import json

import pandas as pd

from automated_support_ticket_classification.config import PROJECT_ROOT
from automated_support_ticket_classification.monitoring.prediction_log import redact

CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "pii_cases.csv"
OUT_PATH = PROJECT_ROOT / "metrics" / "pii_redaction.json"


def score(cases: pd.DataFrame) -> dict:
    cases = cases.fillna("")
    found: dict[str, list[bool]] = {}
    untouched = []
    for row in cases.itertuples(index=False):
        out = redact(row.text)
        if row.kind == "none":
            untouched.append(out == row.text)
            continue
        for value in row.pii.split("|"):
            # Case-insensitive: a lowercased copy of the value is still the value.
            found.setdefault(row.kind, []).append(value.lower() not in out.lower())

    all_found = [hit for hits in found.values() for hit in hits]
    return {
        "pii_values": len(all_found),
        "recall": round(sum(all_found) / len(all_found), 4),
        "recall_by_kind": {k: round(sum(v) / len(v), 4) for k, v in sorted(found.items())},
        "pii_free_messages": len(untouched),
        "untouched": round(sum(untouched) / len(untouched), 4),
    }


def main() -> None:
    report = score(pd.read_csv(CASES_PATH, dtype=str))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
