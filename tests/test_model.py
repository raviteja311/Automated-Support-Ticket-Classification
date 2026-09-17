import pandas as pd

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.models.train import build_pipeline


def test_pipeline_trains_and_predicts():
    cfg = load_config()
    df = pd.DataFrame(
        {
            "text": ["refund my payment", "app crashes error", "reset my password"] * 20,
            "label": ["billing", "technical", "account"] * 20,
        }
    )

    pipe = build_pipeline(cfg)
    pipe.fit(df["text"], df["label"])
    preds = pipe.predict(df["text"])

    assert len(preds) == len(df)
    assert set(preds).issubset({"billing", "technical", "account"})


def test_pipeline_learns_easy_separation():
    cfg = load_config()
    df = pd.DataFrame(
        {
            "text": ["refund payment invoice"] * 30 + ["crash error bug"] * 30,
            "label": ["billing"] * 30 + ["technical"] * 30,
        }
    )

    pipe = build_pipeline(cfg)
    pipe.fit(df["text"], df["label"])

    assert pipe.score(df["text"], df["label"]) > 0.9
