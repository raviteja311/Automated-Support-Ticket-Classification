"""The corpus download retries transient failures instead of failing the stage."""

import time
import urllib.error
import urllib.request

import pytest

from automated_support_ticket_classification.data import banking77


@pytest.fixture
def no_sleep(monkeypatch):
    delays = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


def test_retries_with_backoff_then_succeeds(monkeypatch, tmp_path, no_sleep):
    calls = []

    def flaky(url, target):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.URLError("temporary failure")
        target.write_text("text,category\n", encoding="utf-8")

    monkeypatch.setattr(urllib.request, "urlretrieve", flaky)
    target = tmp_path / "train.csv"
    banking77._fetch("https://example.invalid/train.csv", target)

    assert len(calls) == 3
    assert target.exists()
    # Exponential backoff: each wait doubles.
    assert no_sleep == [banking77._BACKOFF_SECONDS, banking77._BACKOFF_SECONDS * 2]


def test_gives_up_after_the_last_attempt_and_leaves_no_partial_file(
    monkeypatch, tmp_path, no_sleep
):
    def broken(url, target):
        target.write_text("trunc", encoding="utf-8")
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlretrieve", broken)
    target = tmp_path / "train.csv"
    with pytest.raises(urllib.error.URLError):
        banking77._fetch("https://example.invalid/train.csv", target, attempts=3)

    # A truncated file left behind would be mistaken for a cached corpus.
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    assert len(no_sleep) == 2
