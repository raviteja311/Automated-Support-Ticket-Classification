"""Tests for the drift summariser.

Evidently itself is not under test here; its report payload is. The summariser
is the part that decides whether anyone gets woken up, so it is worth pinning
against a realistic payload rather than only against a live run, which is slow
and would make the suite depend on a heavy optional dependency.

The shapes below were taken from an actual `Report.dict()`.
"""

from automated_support_ticket_classification.monitoring import drift


class _FakeResult:
    def __init__(self, payload):
        self._payload = payload

    def dict(self):
        return self._payload


def _payload(label_score, text_score, drifted_count, share):
    return {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "value": {"count": drifted_count, "share": share},
            },
            {
                "metric_name": (
                    "ValueDrift(column=label,method=Jensen-Shannon distance,threshold=0.1)"
                ),
                "value": label_score,
            },
            {
                "metric_name": (
                    "ValueDrift(column=text,method=Absolute text content drift,threshold=0.55)"
                ),
                "value": text_score,
            },
        ]
    }


def test_quiet_when_nothing_has_drifted():
    # Train vs test of one corpus: the correct null result.
    summary = drift.summarise(_FakeResult(_payload(0.0003, 0.5089, 0.0, 0.0)), current_rows=2617)
    assert summary["columns_checked"] == 2
    assert summary["columns_drifted"] == 0
    assert summary["dataset_drifted"] is False
    assert summary["columns"]["label"]["drifted"] is False
    assert summary["columns"]["text"]["drifted"] is False


def test_flags_each_column_against_its_own_threshold():
    # Real corpus vs synthetic templates: both signals move. A detector that
    # never fires is worthless, so this is the case that matters most.
    summary = drift.summarise(_FakeResult(_payload(0.1758, 0.9969, 2.0, 1.0)), current_rows=2617)
    assert summary["columns"]["label"]["drifted"] is True
    assert summary["columns"]["text"]["drifted"] is True
    assert summary["dataset_drifted"] is True


def test_reports_per_column_verdicts_not_just_a_dataset_boolean():
    """Label drift and text drift fail differently and need telling apart.

    Text moving first is the early warning: the input changed before the
    predictions visibly did. A single dataset-level flag hides which it was.
    """
    summary = drift.summarise(_FakeResult(_payload(0.0003, 0.9969, 1.0, 0.5)), current_rows=2617)
    assert summary["columns"]["text"]["drifted"] is True
    assert summary["columns"]["label"]["drifted"] is False
    # share is not strictly greater than the 0.5 threshold, so the dataset as a
    # whole is not flagged even though one column moved.
    assert summary["dataset_drifted"] is False


def test_keeps_the_method_and_threshold_for_context():
    # A score with no threshold beside it is unreadable six months later.
    summary = drift.summarise(_FakeResult(_payload(0.0003, 0.5089, 0.0, 0.0)), current_rows=2617)
    assert summary["columns"]["text"]["threshold"] == 0.55
    assert summary["columns"]["label"]["method"] == "Jensen-Shannon distance"


def test_issues_no_verdict_on_a_thin_sample():
    """The failure this gate exists to stop.

    Five predictions spread evenly across five classes look drifted against a
    training set that is 38% billing and 8% shipping. That is an artifact of
    five being too few to estimate a share from, not a signal. Observed for
    real: a five-row sample reported label drift 0.1764 against a 0.10
    threshold.
    """
    summary = drift.summarise(_FakeResult(_payload(0.1764, 0.9969, 2.0, 1.0)), current_rows=5)
    assert summary["status"] == "insufficient_data"
    assert summary["sufficient_sample"] is False
    # Scores are still reported: they are the evidence, just not enough of it.
    assert summary["columns"]["label"]["score"] == 0.1764
    # But no verdict is issued, and None is not False.
    assert summary["columns"]["label"]["drifted"] is None
    assert summary["dataset_drifted"] is None


def test_issues_a_verdict_once_the_sample_is_big_enough():
    summary = drift.summarise(
        _FakeResult(_payload(0.1764, 0.9969, 2.0, 1.0)),
        current_rows=drift.MIN_SAMPLE_ROWS,
    )
    assert summary["status"] == "ok"
    assert summary["columns"]["label"]["drifted"] is True
    assert summary["dataset_drifted"] is True


def test_unknown_is_not_the_same_as_no_drift():
    """A monitor must be able to tell the two apart.

    Keying an alert on `dataset_drifted` alone would treat an unknown as a
    clean bill of health, which is how a silent monitor happens.
    """
    thin = drift.summarise(_FakeResult(_payload(0.0003, 0.5089, 0.0, 0.0)), current_rows=5)
    clean = drift.summarise(_FakeResult(_payload(0.0003, 0.5089, 0.0, 0.0)), current_rows=2617)
    assert thin["dataset_drifted"] is None
    assert clean["dataset_drifted"] is False
    assert thin["status"] != clean["status"]
