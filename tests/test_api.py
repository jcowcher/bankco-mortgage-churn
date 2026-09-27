import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from churn.model import load_model


@pytest.fixture(scope="module")
def client(trained, features):
    fpath, models_dir = trained
    api_main.state.model, api_main.state.card = load_model(models_dir)
    api_main.state.features = features.set_index("customer_id", drop=False)
    api_main._explain_cached.cache_clear()
    return TestClient(api_main.app)


def test_health_and_model_card(client):
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/model").json()
    assert card["acceptance"]["accepted"] is True


def test_score_returns_propensity_drivers_and_action(client, features):
    cid = int(features["customer_id"].iloc[0])
    r = client.get(f"/score/{cid}")
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_propensity"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}
    assert "action" in body["next_best_action"]
    assert body["model_version"]


def test_score_unknown_customer_is_404(client):
    assert client.get("/score/1").status_code == 404


def test_at_risk_filters_by_region_and_threshold(client):
    r = client.get("/at-risk", params={"region": "North", "threshold": 0.15, "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["region"] == "North"
    assert len(body["customers"]) <= 10
    assert all(c["region"] == "North" and c["churn_propensity"] >= 0.15 for c in body["customers"])
    props = [c["churn_propensity"] for c in body["customers"]]
    assert props == sorted(props, reverse=True)
    assert body["total_annual_value_at_risk"] >= sum(c["annual_value_at_risk"] for c in body["customers"])


def test_at_risk_unknown_region_is_404(client):
    assert client.get("/at-risk", params={"region": "Mars"}).status_code == 404
