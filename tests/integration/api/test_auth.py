import pytest


@pytest.mark.integration
def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.integration
def test_signup_endpoint(client):
    """Test signup endpoint with valid request payload."""
    payload = {
        "temp_token": "some_temp_token",
        "name": "Test User",
        "email": "test_user@example.com"
    }
    response = client.post("/api/v1/auth/signup", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["name"] == "Test User"
    assert data["user"]["email"] == "test_user@example.com"

