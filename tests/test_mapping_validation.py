import pandas as pd
import pytest

from automated_support_ticket_classification.data import mapping_validation as mv
from automated_support_ticket_classification.data.banking77 import INTENT_MAP


def _with_changes(**changes):
    return {**INTENT_MAP, **changes}


def test_sheet_is_blind_alphabetical_and_has_three_examples():
    train = pd.DataFrame(
        {
            "text": [f"msg {i}" for i in range(8)],
            "category": ["card_arrival"] * 4 + ["Refund_not_showing_up"] * 4,
        }
    )
    sheet = mv.build_sheet(train)
    assert list(sheet["intent"]) == ["card_arrival", "Refund_not_showing_up"]
    assert list(sheet.columns) == ["intent", "example_1", "example_2", "example_3", "queue"]
    # Blind: no queue is filled in, so annotators are not anchored to INTENT_MAP.
    assert (sheet["queue"] == "").all()


def test_identical_mappings_agree_perfectly():
    report = mv.agreement({"a": INTENT_MAP, "b": dict(INTENT_MAP), "c": dict(INTENT_MAP)})
    assert report["cohen_kappa"] == {"a_vs_b": 1.0, "a_vs_c": 1.0, "b_vs_c": 1.0}
    assert report["fleiss_kappa"] == 1.0
    assert report["disputed"] == {}


def test_disagreement_lowers_kappa_and_is_listed():
    other = _with_changes(transfer_not_received_by_recipient="billing", cancel_transfer="billing")
    report = mv.agreement({"a": INTENT_MAP, "b": other, "c": dict(INTENT_MAP)})
    assert report["cohen_kappa"]["a_vs_c"] == 1.0
    assert report["cohen_kappa"]["a_vs_b"] < 1.0
    assert report["fleiss_kappa"] < 1.0
    assert set(report["disputed"]) == {"transfer_not_received_by_recipient", "cancel_transfer"}
    assert report["unanimous"] == 75


def test_majority_moves_an_intent_only_when_outvoted():
    b = _with_changes(transfer_not_received_by_recipient="billing", cancel_transfer="billing")
    c = _with_changes(transfer_not_received_by_recipient="billing", cancel_transfer="general")
    majority = mv.majority_map({"author": INTENT_MAP, "b": b, "c": c})
    assert majority["transfer_not_received_by_recipient"] == "billing"
    # account / billing / general: a three-way tie keeps the original.
    assert majority["cancel_transfer"] == INTENT_MAP["cancel_transfer"]


def test_incomplete_sheet_fails_loudly(tmp_path):
    rows = [{"intent": i, "queue": q} for i, q in INTENT_MAP.items()]
    rows[0]["queue"] = ""
    rows[1]["queue"] = "fraud"
    pd.DataFrame(rows).to_csv(tmp_path / "friend.csv", index=False)
    with pytest.raises(ValueError, match="unlabelled intents .* unknown queues \\['fraud'\\]"):
        mv.load_annotations(tmp_path)
