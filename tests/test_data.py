from automated_support_ticket_classification.data.generate import generate
from automated_support_ticket_classification.data.preprocess import clean_text


def test_generate_shape_and_columns():
    df = generate(n_samples=100, seed=0)
    assert len(df) == 100
    assert set(df.columns) == {"text", "label"}


def test_generate_is_reproducible():
    a = generate(50, seed=42)
    b = generate(50, seed=42)
    assert a.equals(b)


def test_clean_text_lowercases_and_collapses_spaces():
    assert clean_text("  Hello   WORLD  ") == "hello world"


def test_generate_covers_all_five_categories():
    # Guards against the page-break transcription bug that dropped the
    # shipping templates and merged technical into account (see E2).
    df = generate(n_samples=500, seed=0)
    assert set(df["label"]) == {"billing", "technical", "account", "shipping", "general"}


def test_banking77_mapping_is_complete_and_uses_the_five_categories():
    """Every banking77 intent maps, and only onto the five known categories.

    Guards the same failure mode as D1 from the other direction: an upstream
    intent appearing with no mapping would silently drop rows.
    """
    from automated_support_ticket_classification.data.banking77 import (
        CATEGORIES,
        INTENT_MAP,
    )

    assert len(INTENT_MAP) == 77
    assert set(INTENT_MAP.values()) == set(CATEGORIES)
    assert set(CATEGORIES) == {"billing", "technical", "account", "shipping", "general"}


def _frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "text": [f"msg {i}" for i in range(10)],
            "label": ["billing", "account"] * 5,
            "split": ["train"] * 7 + ["test"] * 3,
        }
    )


def test_official_split_follows_the_split_column():
    from automated_support_ticket_classification.data.preprocess import split_frame

    train, test = split_frame(_frame(), "official", test_size=0.5, random_state=0)
    # test_size is ignored: the published files decide membership.
    assert (len(train), len(test)) == (7, 3)
    assert "split" not in train.columns and "split" not in test.columns


def test_random_split_ignores_the_split_column():
    from automated_support_ticket_classification.data.preprocess import split_frame

    train, test = split_frame(_frame(), "random", test_size=0.2, random_state=0)
    assert (len(train), len(test)) == (8, 2)
    assert "split" not in train.columns


def test_official_split_without_a_split_column_fails_loudly():
    import pytest

    from automated_support_ticket_classification.data.preprocess import split_frame

    with pytest.raises(ValueError, match="no split column"):
        split_frame(_frame().drop(columns="split"), "official", 0.2, 0)
