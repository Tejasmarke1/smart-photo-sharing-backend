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
def client(mocker):
    """FastAPI test client with dependency overrides."""
    # No setup needed, just return the client
    from fastapi.testclient import TestClient
    return TestClient(app)
