import pytest


@pytest.mark.integration
def test_health_endpoint(client):
    \"\"\"Test health check endpoint.\"\"\"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.integration
def test_signup_endpoint(client):
    \"\"\"Test signup endpoint placeholder.\"\"\"
    response = client.post("/api/v1/auth/signup")
    assert response.status_code == 200
