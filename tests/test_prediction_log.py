"""Tests for prediction-log redaction and rotation, and drift reading it back."""

import json

from automated_support_ticket_classification.monitoring import drift, prediction_log


def test_redacts_emails_and_long_digit_runs():
    text = "card 4111-1111-1111-1111, account 12345678, mail a.b+c@mail.example.co.uk"
    out = prediction_log.redact(text)
    assert out == "card [number], account [number], mail [email]"


def test_keeps_short_numbers_that_carry_routing_signal():
    # Amounts and order numbers identify nobody and help the model route.
    text = "charged $49 twice on order #10231 in 2024"
    assert prediction_log.redact(text) == text


def test_rotates_by_size_and_keeps_only_backup_count(tmp_path):
    log = tmp_path / "predictions.jsonl"
    for i in range(10):
        # Each line is ~30 bytes, so a 70-byte cap holds two lines per file.
        prediction_log.append(log, {"text": f"row {i}", "label": "x"}, 70, backup_count=2)

    assert log.exists()
    backups = prediction_log.rotated_paths(log, 2)
    assert all(p.exists() for p in backups)
    assert not log.with_name(log.name + ".3").exists()
    assert all(p.stat().st_size <= 70 for p in [log, *backups])
    # The newest rows survive; the oldest were rotated out.
    assert "row 9" in log.read_text(encoding="utf-8")
    assert "row 0" not in "".join(p.read_text(encoding="utf-8") for p in [log, *backups])


def test_rotation_disabled_when_max_bytes_is_zero(tmp_path):
    log = tmp_path / "predictions.jsonl"
    for i in range(50):
        prediction_log.append(log, {"text": f"row {i}", "label": "x"}, max_bytes=0)
    assert len(log.read_text(encoding="utf-8").splitlines()) == 50
    assert not log.with_name(log.name + ".1").exists()


def test_drift_reads_rotated_files_oldest_first(tmp_path):
    # A rotation must not shrink the drift sample and silence the verdict.
    log = tmp_path / "predictions.jsonl"
    for i in range(10):
        prediction_log.append(log, {"text": f"row {i}", "label": "x"}, 70, backup_count=5)

    df = drift.load_predictions(log, backup_count=5)
    assert df["text"].tolist() == [f"row {i}" for i in range(10)]


def test_drift_reads_a_redacted_log(tmp_path):
    log = tmp_path / "predictions.jsonl"
    text = prediction_log.redact("refund to jane@example.com please")
    log.write_text(json.dumps({"text": text, "label": "billing"}) + "\n", encoding="utf-8")
    df = drift.load_predictions(log)
    assert list(df.columns) == ["text", "label"]
    assert df["text"][0] == "refund to [email] please"
