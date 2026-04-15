from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_returns_recommendations():
    response = client.get("/researchers/1/dashboard")
    payload = response.json()

    assert response.status_code == 200
    assert payload["researcher"]["researcher_id"] == 1
    assert len(payload["opportunities"]) >= 3
    assert len(payload["collaborators"]) >= 2
    assert payload["analytics"]["ecr_status"] is True


def test_missing_researcher_returns_404():
    response = client.get("/researchers/999/dashboard")
    assert response.status_code == 404
