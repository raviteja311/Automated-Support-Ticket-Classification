"""Tests for prediction-log redaction and rotation, and drift reading it back."""

import json
from pathlib import Path

import pytest

from automated_support_ticket_classification.monitoring import drift, prediction_log


def test_redacts_emails_and_long_digit_runs():
    text = "card 4111-1111-1111-1111, account 12345678, mail a.b+c@mail.example.co.uk"
    out = prediction_log.redact(text)
    assert out == "card [number], account [number], mail [email]"


def test_keeps_short_numbers_that_carry_routing_signal():
    # Amounts and order numbers identify nobody and help the model route.
    text = "charged $49 twice on order #10231 in 2024"
    assert prediction_log.redact(text) == text


def _pii_cases():
    import pandas as pd

    path = Path(__file__).parent / "fixtures" / "pii_cases.csv"
    return pd.read_csv(path, dtype=str).fillna("").itertuples(index=False)


@pytest.mark.parametrize(
    "case", [c for c in _pii_cases() if c.kind != "none"], ids=lambda c: f"{c.kind}: {c.text[:30]}"
)
def test_fixture_pii_is_masked(case):
    out = prediction_log.redact(case.text).lower()
    for value in case.pii.split("|"):
        assert value.lower() not in out, f"{case.kind} {value!r} survived: {out!r}"


@pytest.mark.parametrize(
    "case", [c for c in _pii_cases() if c.kind == "none"], ids=lambda c: f"{c.kind}: {c.text[:30]}"
)
def test_fixture_pii_free_text_is_untouched(case):
    assert prediction_log.redact(case.text) == case.text


def test_email_runs_before_upi():
    # The UPI pattern alone would turn this into [upi].com.
    assert prediction_log.redact("mail ravi@gmail.com") == "mail [email]"


def test_upi_runs_before_long_digits():
    # The digit pattern alone would leave [number]@ybl, keeping the bank handle.
    assert prediction_log.redact("refund to 9876543210@ybl") == "refund to [upi]"


def test_indian_identifiers_get_their_own_tokens():
    text = "pan abcde1234f ifsc SBIN0001234 upi ravi@okaxis"
    assert prediction_log.redact(text) == "pan [pan] ifsc [ifsc] upi [upi]"


def test_aadhaar_and_country_code_mobile_are_masked_whole():
    assert prediction_log.redact("aadhaar 1234 5678 9012") == "aadhaar [number]"
    # The + goes with the number rather than being left as +[number].
    assert prediction_log.redact("call +91 98765 43210") == "call [number]"


def test_card_fragment_keeps_its_context_words():
    # "ending" tells the model this is about a card; only the digits identify.
    assert prediction_log.redact("card ending in 4321") == "card ending in [number]"
    assert prediction_log.redact("card xxxx5678") == "card xxxx[number]"


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
