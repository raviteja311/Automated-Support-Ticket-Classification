"""The prediction log: what the model saw in production, kept safely.

The drift monitor needs the text the API served, so the text has to be written
somewhere. Two things keep that from becoming a liability:

  redaction  emails, UPI IDs, PAN, IFSC codes, long digit runs (card,
             account, Aadhaar and phone numbers) and "card ending 4321"
             fragments are masked before anything reaches disk
  rotation   the file is capped by size and only a few rotated copies are
             kept, so raw traffic is not retained indefinitely

Redaction replaces a value with a placeholder rather than deleting it, so the
text keeps its shape and Evidently's text-drift measure still has something
realistic to compare.

This is a regex baseline. It catches identifiers with a fixed shape. Names and
street addresses have no fixed shape and would need named-entity recognition,
which is out of scope. scripts/pii_recall.py measures what it does catch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EMAIL_TOKEN = "[email]"
UPI_TOKEN = "[upi]"
PAN_TOKEN = "[pan]"
IFSC_TOKEN = "[ifsc]"
NUMBER_TOKEN = "[number]"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# UPI IDs (ravi@okaxis, 9876543210@ybl) look like emails without a dotted
# domain, which is exactly why the email pattern misses them. Run after the
# email pattern, or it would eat the "user@gmail" half of every email.
_UPI = re.compile(r"\b[\w.-]+@[a-z]{2,}\b", re.IGNORECASE)
# PAN: five letters, four digits, one letter (ABCDE1234F). Case-insensitive,
# because customers type it in lower case as often as not.
_PAN = re.compile(r"\b[a-z]{5}\d{4}[a-z]\b", re.IGNORECASE)
# IFSC: four letters, a zero, six letters or digits (SBIN0001234).
_IFSC = re.compile(r"\b[a-z]{4}0[a-z0-9]{6}\b", re.IGNORECASE)
# Eight or more digits, optionally grouped by spaces or dashes: card numbers
# like 4111 1111 1111 1111, account numbers, Aadhaar (1234 5678 9012), phone
# numbers. An optional leading + takes the country code with it, so
# +91 98765 43210 becomes [number], not +[number]. Short runs such as amounts
# or order numbers like #10231 are left alone, because they carry routing
# signal and identify nobody.
_LONG_DIGITS = re.compile(r"\+?\b(?:\d[ -]?){7,}\d\b")
# The last four digits of a card or account. Four digits alone are an amount
# or a year, so they are only masked where the context marks them as a
# fragment: "ending 4321", "ending in 4321", or a masked number like xxxx4321.
_FRAGMENT = re.compile(r"(\bending(?: in| with)? |x{2,} ?)\d{4}\b", re.IGNORECASE)

# Order matters: email before UPI (see _UPI), and UPI before long digits, or
# 9876543210@ybl would become [number]@ybl and leave the handle behind.
_RULES = (
    (_EMAIL, EMAIL_TOKEN),
    (_UPI, UPI_TOKEN),
    (_PAN, PAN_TOKEN),
    (_IFSC, IFSC_TOKEN),
    (_LONG_DIGITS, NUMBER_TOKEN),
    # Keep the context words, mask only the digits.
    (_FRAGMENT, rf"\g<1>{NUMBER_TOKEN}"),
)


def redact(text: str) -> str:
    """Mask emails, UPI IDs, PAN, IFSC, long digit runs and card-number fragments."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


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
