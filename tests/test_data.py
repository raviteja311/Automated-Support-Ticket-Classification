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
