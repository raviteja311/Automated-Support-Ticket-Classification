import json

import pytest
from fastapi.testclient import TestClient

from automated_support_ticket_classification.api import app as app_module
from automated_support_ticket_classification.api.app import app
from automated_support_ticket_classification.api.schemas import MAX_TEXT_LENGTH

client = TestClient(app)


@pytest.fixture(autouse=True)
def scratch_prediction_log(monkeypatch, tmp_path):
    """Keep test requests out of the real log the drift monitor reads."""
    real = app_module.load_config

    def patched():
        cfg = real()
        cfg.monitoring.predictions_path = str(tmp_path / "predictions.jsonl")
        return cfg

    monkeypatch.setattr(app_module, "load_config", patched)


def test_root_points_at_the_docs():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_label_and_confidence():
    response = client.post("/predict", json={"text": "I want a refund for a double charge"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_scores_cover_every_class_and_sum_to_one():
    response = client.post("/predict", json={"text": "my package arrived damaged"})
    scores = response.json()["all_scores"]
    assert set(scores) == {"billing", "technical", "account", "card_delivery", "general"}
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_predict_rejects_empty_text():
    # Rejected by pydantic's min_length=1, before the handler runs.
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_whitespace_only_text():
    # Admitted by pydantic (length 3), rejected by the handler. This is the
    # case that actually exercises the guard in predict().
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_ui_page_renders():
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # The five example buttons are what make the page useful; if the label set
    # ever drifts again (see D1) this catches it in the UI too.
    for label in ("billing", "technical", "account", "card_delivery", "general"):
        assert label in response.text


def test_ui_is_not_in_the_openapi_schema():
    # /ui is a convenience for humans, not part of the documented API contract.
    assert "/ui" not in client.get("/openapi.json").json()["paths"]


def test_predict_accepts_text_at_the_length_limit():
    response = client.post("/predict", json={"text": "a" * MAX_TEXT_LENGTH})
    assert response.status_code == 200


def test_predict_rejects_text_over_the_length_limit():
    # Rejected by pydantic's max_length, before the model ever sees it.
    response = client.post("/predict", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})
    assert response.status_code == 422


def _config_with(monkeypatch, **overrides):
    """Point the app at a modified config for one test."""
    real = app_module.load_config

    def patched():
        cfg = real()
        for dotted, value in overrides.items():
            section, field = dotted.split("__")
            setattr(getattr(cfg, section), field, value)
        return cfg

    monkeypatch.setattr(app_module, "load_config", patched)


def test_missing_model_returns_503_not_500(monkeypatch, tmp_path):
    _config_with(monkeypatch, model__model_path=str(tmp_path / "absent.joblib"))
    app_module.get_model.cache_clear()
    try:
        response = client.post("/predict", json={"text": "my card has not arrived"})
    finally:
        # Drop the failed state so later tests load the real model again.
        app_module.get_model.cache_clear()
    assert response.status_code == 503
    assert "model artifact is missing" in response.json()["detail"]


def test_prediction_log_is_redacted(monkeypatch, tmp_path):
    log = tmp_path / "predictions.jsonl"
    _config_with(monkeypatch, monitoring__predictions_path=str(log))
    text = "Refund to jane.doe@example.com for card 4111 1111 1111 1111 please"
    assert client.post("/predict", json={"text": text}).status_code == 200

    logged = json.loads(log.read_text(encoding="utf-8"))["text"]
    assert "jane.doe@example.com" not in logged
    assert "4111" not in logged
    assert "[email]" in logged and "[number]" in logged


def test_prediction_log_redaction_can_be_switched_off(monkeypatch, tmp_path):
    log = tmp_path / "predictions.jsonl"
    _config_with(monkeypatch, monitoring__predictions_path=str(log), monitoring__redact=False)
    client.post("/predict", json={"text": "write to jane@example.com"})
    assert "jane@example.com" in log.read_text(encoding="utf-8")


def _routed(queue: str) -> float:
    return app_module.ROUTED.labels(queue=queue)._value.get()


def test_confident_prediction_is_routed_to_its_queue(monkeypatch):
    # 0.0 disables the route: every prediction is routed to a queue.
    _config_with(monkeypatch, serve__review_threshold=0.0)
    review_before = _routed("needs_review")
    body = client.post("/predict", json={"text": "I was charged twice this month"}).json()
    assert body["label"] == body["predicted_label"] != "needs_review"
    assert _routed("needs_review") == review_before


def test_unconfident_prediction_goes_to_review_with_the_models_guess(monkeypatch):
    # A threshold above 1.0 sends everything to review.
    _config_with(monkeypatch, serve__review_threshold=1.01)
    before = _routed("needs_review")
    body = client.post("/predict", json={"text": "I was charged twice this month"}).json()
    assert body["label"] == "needs_review"
    assert body["predicted_label"] in body["all_scores"]
    assert body["confidence"] == body["all_scores"][body["predicted_label"]]
    assert _routed("needs_review") == before + 1


def test_review_counter_is_exposed_to_prometheus():
    client.post("/predict", json={"text": "I was charged twice this month"})
    assert "ticket_routed_total" in client.get("/metrics").text
