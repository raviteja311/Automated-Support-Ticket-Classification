"""Tests for the promotion gate.

The gate is pure decision logic, so it is tested without touching MLflow:
`production_metric` and `promote` are the seams, and both are stubbed. What
matters is the rule, not the plumbing.
"""

from automated_support_ticket_classification.models import registry


def _capture(monkeypatch, incumbent, incumbent_source="synthetic"):
    """Stub the registry so the gate can be tested in isolation.

    Returns a list that records any version that gets promoted.
    """
    promoted: list[str] = []
    monkeypatch.setattr(
        registry, "production_metric", lambda metric=None: (incumbent, incumbent_source)
    )
    monkeypatch.setattr(registry, "promote", lambda version: promoted.append(version))
    return promoted


def test_promotes_when_nothing_is_in_production(monkeypatch):
    promoted = _capture(monkeypatch, incumbent=None, incumbent_source=None)
    assert registry.promote_if_better("1", 0.50, data_source="synthetic") is True
    assert promoted == ["1"]


def test_promotes_when_candidate_is_better(monkeypatch):
    promoted = _capture(monkeypatch, incumbent=0.90)
    assert registry.promote_if_better("7", 0.93, data_source="synthetic") is True
    assert promoted == ["7"]


def test_does_not_promote_on_a_tie(monkeypatch):
    # A tie must not ship. The incumbent already works; an equal model costs a
    # deploy and gains nothing.
    promoted = _capture(monkeypatch, incumbent=0.9297)
    assert registry.promote_if_better("2", 0.9297, data_source="synthetic") is False
    assert promoted == []


def test_does_not_promote_a_regression(monkeypatch):
    promoted = _capture(monkeypatch, incumbent=0.93)
    assert registry.promote_if_better("3", 0.88, data_source="synthetic") is False
    assert promoted == []


def test_promotes_across_a_corpus_change_even_when_the_number_is_lower(monkeypatch):
    """The case that exposed the flaw in naive metric gating.

    A model trained on real data scored 0.9171 against an incumbent trained on
    synthetic data scoring 0.9297. The gate refused, but the comparison was
    meaningless: different test sets are not the same scale. When the corpus
    changes the incumbent is not evidence of anything.
    """
    promoted = _capture(monkeypatch, incumbent=0.9297, incumbent_source="synthetic")
    assert registry.promote_if_better("4", 0.9171, data_source="banking77") is True
    assert promoted == ["4"]


def test_still_compares_normally_within_one_corpus(monkeypatch):
    # The corpus escape hatch must not become a way to ship anything at all.
    promoted = _capture(monkeypatch, incumbent=0.9297, incumbent_source="banking77")
    assert registry.promote_if_better("5", 0.9171, data_source="banking77") is False
    assert promoted == []


def test_gate_metric_is_macro_f1_not_accuracy():
    # Accuracy can rise while a minority class collapses, which is exactly how
    # the four-class data defect hid for so long. Gate on macro F1.
    assert registry.GATE_METRIC == "f1_macro"
