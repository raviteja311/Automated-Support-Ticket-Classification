"""The prediction log: what the model saw in production, kept safely.

The drift monitor needs the text the API served, so the text has to be written
somewhere. Two things keep that from becoming a liability:

  redaction  emails and long digit runs (card and account numbers) are masked
             before anything reaches disk
  rotation   the file is capped by size and only a few rotated copies are
             kept, so raw traffic is not retained indefinitely

Redaction replaces a value with a placeholder rather than deleting it, so the
text keeps its shape and Evidently's text-drift measure still has something
realistic to compare.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EMAIL_TOKEN = "[email]"
NUMBER_TOKEN = "[number]"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Eight or more digits, optionally grouped by spaces or dashes: card numbers
# like 4111 1111 1111 1111, IBAN-style account numbers, phone numbers. Short
# runs such as amounts or order numbers like #10231 are left alone, because
# they carry routing signal and identify nobody.
_LONG_DIGITS = re.compile(r"\b(?:\d[ -]?){7,}\d\b")


def redact(text: str) -> str:
    """Mask emails and long digit runs."""
    text = _EMAIL.sub(EMAIL_TOKEN, text)
    return _LONG_DIGITS.sub(NUMBER_TOKEN, text)


def rotated_paths(path: Path, backup_count: int) -> list[Path]:
    """Return the rotated siblings of `path`, newest first: .1, .2, ..."""
    return [path.with_name(f"{path.name}.{i}") for i in range(1, backup_count + 1)]


def _rotate(path: Path, backup_count: int) -> None:
    """Shift path -> path.1 -> path.2 ..., dropping whatever falls off the end."""
    backups = rotated_paths(path, backup_count)
    if not backups:
        # Rotation with no backups kept is simply truncation.
        path.unlink(missing_ok=True)
        return
    backups[-1].unlink(missing_ok=True)
    for older, newer in zip(reversed(backups[1:]), reversed(backups[:-1]), strict=True):
        if newer.exists():
            newer.replace(older)
    path.replace(backups[0])


def append(
    path: Path,
    record: dict,
    max_bytes: int = 0,
    backup_count: int = 0,
) -> None:
    """Append one JSON line, rotating first if it would push the file past max_bytes."""
    line = json.dumps(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_bytes > 0 and path.exists():
        if path.stat().st_size + len(line.encode("utf-8")) > max_bytes:
            _rotate(path, backup_count)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
