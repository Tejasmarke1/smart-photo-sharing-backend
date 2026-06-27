"""
Integration test configuration
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock Redis before any app imports
mock_redis = MagicMock()
mock_redis.from_url.return_value = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['redis.asyncio'] = mock_redis

# Now we can safely import the app
from src.app.main import app


@pytest.fixture
def client(db):
    """FastAPI test client with database dependency overrides."""
    from src.db.base import get_db
    from fastapi.testclient import TestClient
    
    def override_get_db():
        try:
            yield db
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

