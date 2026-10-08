"""Spec for models/granularity.py. Delete the xfail marker once implemented."""

import pandas as pd
import pytest

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.models.granularity import granularity_scores

pending = pytest.mark.xfail(raises=NotImplementedError, reason="stub: implement granularity.py")

# Two intents that share a queue, plus three that do not.
PHRASES = {
    ("card_arrival", "shipping"): "when will my new card arrive in the post",
    ("card_delivery_estimate", "shipping"): "how long does card delivery usually take",
    ("declined_card_payment", "technical"): "my card payment was declined at the shop",
    ("exchange_rate", "billing"): "what exchange rate do you charge for euros",
    ("change_pin", "account"): "how do i change the pin on my card",
}


def _frame(n_per_intent: int) -> pd.DataFrame:
    rows = [
        {"text": f"{phrase} {i}", "label": label, "intent": intent}
        for (intent, label), phrase in PHRASES.items()
        for i in range(n_per_intent)
    ]
    return pd.DataFrame(rows)


@pending
def test_returns_both_rows_with_scores_in_range():
    scores = granularity_scores(_frame(8), _frame(2), load_config())
    assert set(scores) == {"77way", "77way_mapped"}
    for row in scores.values():
        assert set(row) == {"accuracy", "f1_macro"}
        assert all(0.0 <= v <= 1.0 for v in row.values())


@pending
def test_mapping_never_lowers_accuracy():
    # A correct intent always maps to the correct queue, and a wrong intent can
    # still land in the right queue, so mapped accuracy can only be higher.
    scores = granularity_scores(_frame(8), _frame(2), load_config())
    assert scores["77way_mapped"]["accuracy"] >= scores["77way"]["accuracy"]
