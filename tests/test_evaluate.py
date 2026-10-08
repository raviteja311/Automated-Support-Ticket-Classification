import numpy as np
import pandas as pd

from automated_support_ticket_classification.models.evaluate import (
    confusion_frame,
    errors_frame,
)


def test_confusion_frame_rows_are_true_and_columns_are_predicted():
    cm = confusion_frame(
        ["billing", "billing", "account"],
        ["billing", "account", "account"],
        labels=["account", "billing"],
    )
    assert cm.loc["billing", "account"] == 1  # one billing row predicted as account
    assert cm.loc["account", "billing"] == 0
    assert int(cm.to_numpy().sum()) == 3


def test_errors_frame_keeps_only_mistakes_least_confident_first():
    test_df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "label": ["billing", "account", "general"],
            "intent": ["card_payment_fee_charged", "edit_personal_details", "atm_support"],
        }
    )
    errors = errors_frame(
        test_df,
        preds=np.array(["billing", "billing", "technical"]),
        confidence=np.array([0.9, 0.8, 0.4]),
    )
    assert list(errors["text"]) == ["c", "b"]
    assert list(errors.columns) == [
        "text",
        "true_queue",
        "predicted_queue",
        "confidence",
        "intent",
    ]


def test_errors_frame_without_intent_column():
    # Synthetic data has no 77-way intent.
    test_df = pd.DataFrame({"text": ["a"], "label": ["billing"]})
    errors = errors_frame(test_df, np.array(["account"]), np.array([0.5]))
    assert "intent" not in errors.columns
