"""Map the banking77 corpus onto this project's five support categories.

banking77 is 13,000 real customer-support messages labelled with 77 fine
grained intents. This module folds those intents into the five categories the
service classifies, so the project can train on human-written text instead of
templates.

Why this corpus. The obvious candidate, the CFPB consumer-complaints database,
turns out to publish no complaint narrative in either its bulk export or its
API, so there is no text to classify. banking77 is real support traffic, it is
1 MB rather than 347 MB, and its intents cover all five categories including
card delivery, which is what makes a `shipping` class possible at all.

About the mapping. It is a judgement, not a ground truth. The rule applied
throughout is *what would a support team need to do about this*, which is what
a routing model is for:

  billing    money moved wrongly, or a charge needs explaining or reversing
  technical  something did not work: declined, failed, unrecognised by a device
  account    identity, credentials, limits, and the account's lifecycle
  shipping   a physical card needs to arrive
  general    an informational question with no fault to fix

Borderline calls are marked below. They are the honest cost of reusing a corpus
built for a different taxonomy, and a reviewer is entitled to disagree with any
of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)

# Intent -> category. Every one of banking77's 77 intents appears exactly once;
# INTENT_MAP completeness is enforced by a test.
INTENT_MAP: dict[str, str] = {
    # ---- billing: money moved wrongly, or a charge needs explaining ----
    "Refund_not_showing_up": "billing",
    "request_refund": "billing",
    "card_payment_fee_charged": "billing",
    "card_payment_not_recognised": "billing",
    "card_payment_wrong_exchange_rate": "billing",
    "cash_withdrawal_charge": "billing",
    "cash_withdrawal_not_recognised": "billing",
    "direct_debit_payment_not_recognised": "billing",
    "exchange_charge": "billing",
    "exchange_rate": "billing",
    "extra_charge_on_statement": "billing",
    "reverted_card_payment?": "billing",
    "top_up_by_bank_transfer_charge": "billing",
    "top_up_by_card_charge": "billing",
    "transaction_charged_twice": "billing",
    "transfer_fee_charged": "billing",
    "wrong_amount_of_cash_received": "billing",
    "wrong_exchange_rate_for_cash_withdrawal": "billing",
    "top_up_reverted": "billing",
    # Borderline: "pending" and "balance not updated" are status questions about
    # money, not faults. Billing, because the answer is about a transaction.
    "pending_card_payment": "billing",
    "pending_cash_withdrawal": "billing",
    "pending_top_up": "billing",
    "pending_transfer": "billing",
    "balance_not_updated_after_bank_transfer": "billing",
    "balance_not_updated_after_cheque_or_cash_deposit": "billing",
    # ---- technical: it did not work ----
    "card_not_working": "technical",
    "contactless_not_working": "technical",
    "virtual_card_not_working": "technical",
    "declined_card_payment": "technical",
    "declined_cash_withdrawal": "technical",
    "declined_transfer": "technical",
    "failed_transfer": "technical",
    "top_up_failed": "technical",
    "card_swallowed": "technical",
    "apple_pay_or_google_pay": "technical",
    "exchange_via_app": "technical",
    "card_linking": "technical",
    "atm_support": "technical",
    # Borderline: a blocked beneficiary is a rule rejecting an action, which
    # reads to the user as a failure rather than a policy question.
    "beneficiary_not_allowed": "technical",
    # ---- account: identity, credentials, limits, lifecycle ----
    "activate_my_card": "account",
    "age_limit": "account",
    "change_pin": "account",
    "passcode_forgotten": "account",
    "pin_blocked": "account",
    "edit_personal_details": "account",
    "terminate_account": "account",
    "unable_to_verify_identity": "account",
    "verify_my_identity": "account",
    "verify_source_of_funds": "account",
    "verify_top_up": "account",
    "why_verify_identity": "account",
    "compromised_card": "account",
    "lost_or_stolen_card": "account",
    "lost_or_stolen_phone": "account",
    "disposable_card_limits": "account",
    "top_up_limits": "account",
    "automatic_top_up": "account",
    "cancel_transfer": "account",
    # ---- shipping: a physical card needs to arrive ----
    "card_arrival": "shipping",
    "card_delivery_estimate": "shipping",
    "get_physical_card": "shipping",
    "getting_spare_card": "shipping",
    "order_physical_card": "shipping",
    # Borderline: an expiring card is a status question, but resolving it means
    # posting a replacement, so it routes to the same team.
    "card_about_to_expire": "shipping",
    # ---- general: informational, nothing to fix ----
    "country_support": "general",
    "fiat_currency_support": "general",
    "supported_cards_and_currencies": "general",
    "visa_or_mastercard": "general",
    "card_acceptance": "general",
    "transfer_timing": "general",
    "receiving_money": "general",
    "transfer_into_account": "general",
    "top_up_by_cash_or_cheque": "general",
    "topping_up_by_card": "general",
    "get_disposable_virtual_card": "general",
    "getting_virtual_card": "general",
    "transfer_not_received_by_recipient": "general",
}

CATEGORIES = ("billing", "technical", "account", "shipping", "general")


def map_intent(intent: str) -> str | None:
    """Return the category for a banking77 intent, or None if unmapped."""
    return INTENT_MAP.get(intent)


_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
_FILES = ("train.csv", "test.csv")


def _download(cache_dir: Path) -> None:
    """Fetch the corpus once into `cache_dir`.

    Roughly 1 MB over two files. Cached because a training stage that hits the
    network on every run is neither fast nor reproducible.
    """
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        target = cache_dir / name
        if target.exists():
            continue
        logger.info("Downloading %s/%s", _BASE, name)
        urllib.request.urlretrieve(f"{_BASE}/{name}", target)  # noqa: S310


def load(cache_dir: Path, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Return banking77 as a text/label frame using this project's categories.

    Downloads on first use, then reads from the cache. The natural class
    balance is kept rather than resampled: real support traffic is skewed, and
    flattening it would hide exactly the effect macro F1 exists to measure.
    """
    _download(cache_dir)

    frames = [pd.read_csv(cache_dir / name) for name in _FILES]
    df = pd.concat(frames, ignore_index=True)

    df["label"] = df["category"].map(INTENT_MAP)
    unmapped = int(df["label"].isna().sum())
    if unmapped:
        # Every intent is mapped and a test enforces that, so this means the
        # upstream corpus gained an intent. Loud, not silent.
        raise ValueError(f"{unmapped} rows have intents missing from INTENT_MAP")

    df = df[["text", "label"]]
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed)

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
