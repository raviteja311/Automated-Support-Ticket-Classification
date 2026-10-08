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


def test_calibrated_svc_gives_probabilities():
    """The reason to calibrate: the API needs predict_proba for confidence."""
    import pandas as pd

    from automated_support_ticket_classification.config import load_config

    cfg = load_config()
    texts = ["refund my payment"] * 10 + ["app keeps crashing"] * 10
    labels = ["billing"] * 10 + ["technical"] * 10
    pipe = build_pipeline(cfg, "linearsvc_calibrated").fit(pd.Series(texts), labels)
    proba = pipe.predict_proba(["refund please"])[0]
    assert abs(proba.sum() - 1.0) < 1e-6
    assert list(pipe.classes_) == ["billing", "technical"]


def test_unknown_model_type_fails_loudly():
    import pytest

    from automated_support_ticket_classification.config import load_config

    with pytest.raises(ValueError, match="Unknown model.type"):
        build_pipeline(load_config(), "random_forest")
