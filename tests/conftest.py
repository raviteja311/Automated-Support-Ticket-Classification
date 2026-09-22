from pathlib import Path

import joblib
import pandas as pd
import pytest

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.models.train import build_pipeline


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    """Guarantee a model file exists so API tests can load it (local and CI).

    CI never runs the pipeline, so no artifact exists there. Building a tiny
    one on demand keeps the suite self-sufficient. The label set matches the
    real pipeline's five classes; if it did not, the API tests would exercise
    a model that production could never produce.
    """
    cfg = load_config()
    path = Path(cfg.model.model_path)
    if not path.exists():
        df = pd.DataFrame(
            {
                "text": (
                    ["refund my payment invoice"] * 10
                    + ["app crashes with an error"] * 10
                    + ["reset my password please"] * 10
                    + ["where is my order shipping"] * 10
                    + ["what are your support hours"] * 10
                ),
                "label": (
                    ["billing"] * 10
                    + ["technical"] * 10
                    + ["account"] * 10
                    + ["shipping"] * 10
                    + ["general"] * 10
                ),
            }
        )
        pipe = build_pipeline(cfg)
        pipe.fit(df["text"], df["label"])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, path)
    yield
