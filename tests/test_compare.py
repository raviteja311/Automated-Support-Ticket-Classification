"""Tests for the calibration functions and the rule check in models/compare.py."""

import numpy as np
import pytest

from automated_support_ticket_classification.models import compare


def test_reliability_table_bins_by_confidence():
    confidence = np.array([0.15, 0.15, 0.85, 0.85, 0.85, 0.85])
    correct = np.array([False, True, True, True, True, False])
    table = compare.reliability_table(confidence, correct, n_bins=10)
    # Two non-empty bins: [0.1, 0.2) and [0.8, 0.9).
    assert list(table["count"]) == [2, 4]
    assert list(table["accuracy"]) == [0.5, 0.75]
    assert np.allclose(table["mean_confidence"], [0.15, 0.85])
    assert np.allclose(table["bin_low"], [0.1, 0.8])


def test_confidence_of_exactly_one_lands_in_the_last_bin():
    table = compare.reliability_table(np.array([1.0]), np.array([True]), n_bins=10)
    assert np.allclose(table["bin_high"], [1.0])


def test_ece_is_zero_when_confidence_matches_accuracy():
    # 80% confident, right 4 times out of 5.
    confidence = np.full(5, 0.8)
    correct = np.array([True, True, True, True, False])
    assert compare.expected_calibration_error(confidence, correct) == 0.0


def test_ece_measures_overconfidence():
    # 90% confident, right half the time: off by 0.4 everywhere.
    confidence = np.full(10, 0.9)
    correct = np.array([True, False] * 5)
    assert compare.expected_calibration_error(confidence, correct) == pytest.approx(0.4)


def test_ece_weights_bins_by_count():
    # 8 perfectly calibrated rows at 0.5 and 2 rows at 0.95 that are all wrong.
    confidence = np.array([0.5] * 8 + [0.95] * 2)
    correct = np.array([True, False] * 4 + [False, False])
    # (8 * 0 + 2 * 0.95) / 10
    assert compare.expected_calibration_error(confidence, correct) == pytest.approx(0.19)


def test_comparison_refuses_until_the_rule_is_written(tmp_path):
    rule = tmp_path / "model-decision.md"
    rule.write_text("## Rule\n\nTODO: write it\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="decision rule"):
        compare.check_rule(rule)

    rule.write_text("## Rule\n\nKeep LogReg unless SVC wins with p < 0.05.\n", encoding="utf-8")
    assert len(compare.check_rule(rule)) == 64


def test_a_confidence_on_a_bin_edge_goes_in_the_bin_it_opens():
    # np.linspace(0, 1, 11)[7] is 0.7000000000000001; 0.7 must still open [0.7, 0.8).
    table = compare.reliability_table(np.array([0.7]), np.array([True]), n_bins=10)
    assert np.allclose(table["bin_low"], [0.7])
