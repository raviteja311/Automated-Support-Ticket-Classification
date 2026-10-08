import pandas as pd
import pytest

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.models import eval_india
from automated_support_ticket_classification.models.train import build_pipeline


def _rows(*rows):
    return pd.DataFrame(rows, columns=eval_india.COLUMNS)


def test_validate_normalises_case_and_whitespace():
    df = eval_india.validate(_rows(("upi failed", " Billing ", "Hinglish", "me")))
    assert (df["label"].iloc[0], df["language_style"].iloc[0]) == ("billing", "hinglish")


def test_validate_reports_every_bad_row_with_its_line_number():
    df = _rows(
        ("ok", "billing", "english", "me"),
        ("", "shipping", "tamil", "me"),
    )
    with pytest.raises(ValueError) as exc:
        eval_india.validate(df)
    message = str(exc.value)
    assert "line 3: empty text" in message
    assert "line 3: unknown label 'shipping'" in message
    assert "line 3: unknown language_style 'tamil'" in message


def test_evaluate_scores_overall_and_by_style():
    texts = ["refund my payment"] * 10 + ["app keeps crashing"] * 10
    labels = ["billing"] * 10 + ["technical"] * 10
    model = build_pipeline(load_config()).fit(pd.Series(texts), labels)
    df = eval_india.validate(
        _rows(
            ("refund my payment please", "billing", "english", "a"),
            ("app crashing", "technical", "short_or_misspelled", "a"),
            ("app keeps crashing", "billing", "hinglish", "b"),
        )
    )
    report, errors = eval_india.evaluate(model, df)
    assert report["overall"]["rows"] == 3
    assert set(report["by_style"]) == {"english", "hinglish", "short_or_misspelled"}
    assert report["by_style"]["english"]["accuracy"] == 1.0
    # The mislabelled-on-purpose row is the one error, with its prediction.
    assert list(errors["text"]) == ["app keeps crashing"]
    assert errors["predicted"].iloc[0] == "technical"
