from fastapi.testclient import TestClient

from automated_support_ticket_classification.api.app import app

client = TestClient(app)


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
    assert set(scores) == {"billing", "technical", "account", "shipping", "general"}
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
