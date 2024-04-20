from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
sys.path.append('../.')
from src.app import app

client = TestClient(app)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# Test the health check endpoint
def test_health_check(client):
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test the date check endpoint
def test_date_check(client):
    response = client.get("/datecheck")
    assert response.status_code == 200
    assert "date" in response.json()

# Mocking the summarizer to test summarization endpoint
from unittest.mock import patch

@pytest.mark.parametrize("text, content_type, expected_status, expected_response", [
    ("This is a test text for a job", "job", 200, {"summary": "mocked summary"}),
    ("Invalid content type text", "invalid_type", 400, {"detail": "content_type must be one of ['job', 'course', 'scholarship']"}),
    ("", "job", 400, {"detail": "Text must not be empty"}),
])
async def test_create_summary(client, text, content_type, expected_status, expected_response):
    with patch('your_module_path.py.summarizer.text_summarize', return_value="mocked summary"):
        response = client.post("/summarize/", json={"text": text, "content_type": content_type})
        assert response.status_code == expected_status
        assert response.json() == expected_response

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_async_create_summary(async_client):
    # Example of an async test
    response = await async_client.post("/summarize/", json={"text": "Async test text", "content_type": "job"})
    assert response.status_code == 200
    assert "summary" in response.json()


# 
# python -m pytest -p no:warnings
