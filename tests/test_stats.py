"""Spec for models/stats.py.

The xfail markers keep the suite green while the functions are stubs. They
only absorb NotImplementedError, so once a function is written any wrong
answer fails for real. Delete the marker when you implement each function.
"""

import numpy as np
import pytest

from automated_support_ticket_classification.models.stats import bootstrap_f1_ci, mcnemar_p

pending = pytest.mark.xfail(raises=NotImplementedError, reason="stub: implement in stats.py")

LABELS = np.array(["billing", "account", "general", "technical"] * 50)


def _noisy(y, wrong_every: int):
    """Copy of y with every nth label replaced by a different one."""
    y = y.copy()
    y[::wrong_every] = np.where(y[::wrong_every] == "billing", "account", "billing")
    return y


@pending
def test_bootstrap_ci_of_perfect_predictions_is_a_point():
    assert bootstrap_f1_ci(LABELS, LABELS) == (1.0, 1.0)


@pending
def test_bootstrap_ci_brackets_the_point_estimate():
    from sklearn.metrics import f1_score

    pred = _noisy(LABELS, 5)
    low, high = bootstrap_f1_ci(LABELS, pred)
    assert low < f1_score(LABELS, pred, average="macro") < high
    assert 0.0 <= low < high <= 1.0


@pending
def test_bootstrap_ci_is_reproducible_by_seed():
    pred = _noisy(LABELS, 5)
    assert bootstrap_f1_ci(LABELS, pred, seed=1) == bootstrap_f1_ci(LABELS, pred, seed=1)


@pending
def test_bootstrap_ci_narrows_with_a_lower_confidence_level():
    pred = _noisy(LABELS, 5)
    low95, high95 = bootstrap_f1_ci(LABELS, pred, alpha=0.05)
    low50, high50 = bootstrap_f1_ci(LABELS, pred, alpha=0.5)
    assert high50 - low50 < high95 - low95


@pending
def test_mcnemar_detects_one_model_fixing_the_others_errors():
    # B gets 40 rows wrong that A gets right; nowhere does B beat A.
    pred_b = _noisy(LABELS, 5)
    p = mcnemar_p(LABELS, LABELS, pred_b)
    assert p < 0.001


@pending
def test_mcnemar_sees_no_difference_when_errors_trade_evenly():
    # Each model is wrong on 20 rows the other gets right.
    pred_a, pred_b = LABELS.copy(), LABELS.copy()
    pred_a[0:80:4] = "general"
    pred_b[1:80:4] = "billing"
    assert mcnemar_p(LABELS, pred_a, pred_b) > 0.5


@pending
def test_mcnemar_is_symmetric_in_the_two_models():
    pred_a, pred_b = _noisy(LABELS, 5), _noisy(LABELS, 7)
    assert mcnemar_p(LABELS, pred_a, pred_b) == mcnemar_p(LABELS, pred_b, pred_a)
