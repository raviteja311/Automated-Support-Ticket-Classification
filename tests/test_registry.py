"""Tests for the promotion gate.

The gate is pure decision logic, so it is tested without touching MLflow:
`production_metric` and `promote` are the seams, and both are stubbed. What
matters is the rule, not the plumbing.
"""

from automated_support_ticket_classification.models import registry


def _capture(monkeypatch, incumbent):
    """Stub the registry so the gate can be tested in isolation.

    Returns a list that records any version that gets promoted.
    """
    promoted: list[str] = []
    monkeypatch.setattr(registry, "production_metric", lambda metric=None: incumbent)
    monkeypatch.setattr(registry, "promote", lambda version: promoted.append(version))
    return promoted


def test_promotes_when_nothing_is_in_production(monkeypatch):
    promoted = _capture(monkeypatch, incumbent=None)
    assert registry.promote_if_better("1", candidate_metric=0.50) is True
    assert promoted == ["1"]


def test_promotes_when_candidate_is_better(monkeypatch):
    promoted = _capture(monkeypatch, incumbent=0.90)
    assert registry.promote_if_better("7", candidate_metric=0.93) is True
    assert promoted == ["7"]


def test_does_not_promote_on_a_tie(monkeypatch):
    # A tie must not ship. The incumbent already works; an equal model costs a
    # deploy and gains nothing.
    promoted = _capture(monkeypatch, incumbent=0.9297)
    assert registry.promote_if_better("2", candidate_metric=0.9297) is False
    assert promoted == []


def test_does_not_promote_a_regression(monkeypatch):
    promoted = _capture(monkeypatch, incumbent=0.93)
    assert registry.promote_if_better("3", candidate_metric=0.88) is False
    assert promoted == []


def test_gate_metric_is_macro_f1_not_accuracy():
    # Accuracy can rise while a minority class collapses, which is exactly how
    # the four-class data defect hid for so long. Gate on macro F1.
    assert registry.GATE_METRIC == "f1_macro"
