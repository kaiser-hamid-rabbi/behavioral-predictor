import pytest
from fastapi.testclient import TestClient

from app.main import app

from app.core.dependencies import get_db_session, get_redis

async def get_mock_db_session():
    class MockSession:
        async def commit(self): pass
        async def rollback(self): pass
        async def execute(self, *args, **kwargs): 
            class MockResult:
                def scalars(self): 
                    class MockScalars:
                        def first(self): return None
                        def all(self): return []
                    return MockScalars()
            return MockResult()
    yield MockSession()

# Since our depends functions rely on db being up we can mock them
app.dependency_overrides[get_db_session] = get_mock_db_session
app.dependency_overrides[get_redis] = lambda: None

client = TestClient(app)

def test_health_check_returns_schema():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "components" in data

def test_ingest_event_batch_rejected_if_empty():
    response = client.post("/events/ingest", json={"events": []})
    assert response.status_code == 422 # Validation error

def test_predict_endpoint_with_no_history():
    payload = {
        "user_id": "00000000-0000-0000-0000-000000000000",
        "events": [
            {
                "event_id": "11111111-1111-1111-1111-111111111111",
                "muid": "00000000-0000-0000-0000-000000000000",
                "event_name": "pageview",
                "event_time": "2024-01-01T10:00:00Z",
                "device_os": "desktop",
                "channel": "browser",
                "traffic_source": "direct",
                "category": "home"
            }
        ]
    }
    response = client.post("/predict", json=payload)
    # This should fail with 503 because ModelNotReadyError might be triggered
    # since we have no active model deployed in test db.
    assert response.status_code == 503

