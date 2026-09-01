import random
from pathlib import Path

import pandas as pd

from ticket_triage.config import load_config
from ticket_triage.logger import get_logger

logger = get_logger(__name__)

TEMPLATES = {
    "billing": [
    "I was charged {amount} twice on my {card} card this month.",
    "Why is my invoice higher than the {plan} plan price?",
    "I need a refund for the duplicate payment of {amount}.",
    "My subscription renewed but I wanted to cancel the billing.",
    "There is an unexpected fee of {amount} on my latest bill.",
    ],

    "technical": [
    "The application is running slow and freezing frequently.",
    "I am experiencing issues with the {feature} functionality.",
    "The API is returning a {code} error when I make a request.",
    "There is a bug in the {feature} that needs to be fixed.",
    "The software is not compatible with my operating system.",
    ],
    
    "account": [
    "The app crashes every time I open the {feature} page.",
    "I get an error code {code} when I try to log in.",
    "The dashboard is not loading after the latest update.",
    "Sync keeps failing between my phone and the web {feature}.",
    "The export button does nothing and throws error {code}.",
    "I cannot reset my password, the email never arrives.",
    "Please help me change the email address on my account.",
    "My account is locked after too many login attempts.",
    "I want to delete my account and all of my data.",
    "How do I enable two factor authentication on my profile?",
    ],
    


    "general": [
        "Do you offer a discount for students or non profits?",
        "What are your customer support working hours?",
        "I would like to share some feedback about your service.",
        "Is there a mobile app available for {feature}?",
        "Can you tell me more about the {plan} plan features?",
    ],
}

FILLERS = {
    "amount": ["$9.99", "$19.99", "$49", "$120", "$5"],
    "card": ["Visa", "Mastercard", "Amex"],
    "plan": ["Basic", "Pro", "Enterprise"],
    "feature": ["reports", "billing", "settings", "calendar"],
    "code": ["500", "403", "TIMEOUT", "NULL_REF"],
    "order": ["#10231", "#55012", "#77341", "#90210"],
}

def _fill(template: str) -> str:
    text = template
    for key, choices in FILLERS.items():
        token = "{" + key + "}"
        if token in text:
            text = text.replace(token, random.choice(choices))
    return text

def generate(n_samples: int, seed: int) -> pd.DataFrame:
    """Create a reproducible synthetic support-ticket dataset."""
    random.seed(seed)
    labels = list(TEMPLATES.keys())
    rows = []
    for _ in range(n_samples):
        label = random.choice(labels)
        text = _fill(random.choice(TEMPLATES[label]))
        # 10% label noise keeps the classification task non-trivial
        stored_label = random.choice(labels) if random.random() < 0.10 else label
        rows.append({"text": text, "label": stored_label})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = generate(cfg.data.n_samples, cfg.data.random_state)
    out = Path(cfg.data.raw_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    main()