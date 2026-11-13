import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.main import create_app


@pytest.fixture(scope="module")
def client():
    """Start FastAPI app once for the test module so lifespan loads the ML model."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    settings = get_settings()
    return {"X-API-Key": settings.api_key}


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert data["is_synthetic_data"] is True
    assert data["decision_threshold"] == 0.45
    assert len(data["features"]) == 6
    assert "pr_auc" in data["test_metrics"]


def test_predict_valid_normal_input(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    payload = {
        "type": "L",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["failure_class"] == 0
    assert 0.0 <= data["failure_probability"] < 0.45
    assert data["model_version"] == "1.0.0"
    assert "synthetic" in data["disclaimer"].lower()


def test_predict_valid_high_risk_input(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    payload = {
        "type": "L",
        "air_temperature_k": 304.0,
        "process_temperature_k": 313.2,
        "rotational_speed_rpm": 1350,
        "torque_nm": 68.5,
        "tool_wear_min": 220,
    }
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["failure_class"] == 1
    assert 0.45 <= data["failure_probability"] <= 1.0
    assert data["model_version"] == "1.0.0"


def test_predict_invalid_input_returns_consistent_error(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    invalid_payload = {
        "type": "INVALID_TYPE",
        "air_temperature_k": -50.0,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }
    response = client.post("/predict", json=invalid_payload, headers=auth_headers)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(data["error"]["details"], list)
    assert len(data["error"]["details"]) >= 2


def test_predict_missing_api_key_returns_401(client: TestClient) -> None:
    payload = {
        "type": "L",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "UNAUTHORIZED"


def test_predict_wrong_api_key_returns_401(client: TestClient) -> None:
    payload = {
        "type": "L",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }
    response = client.post(
        "/predict", json=payload, headers={"X-API-Key": "definitely-wrong-key"}
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "UNAUTHORIZED"


def test_cors_enforces_configured_origin(client: TestClient) -> None:
    settings = get_settings()
    allowed = client.options(
        "/predict",
        headers={
            "Origin": settings.allowed_origin,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == settings.allowed_origin

    blocked = client.options(
        "/predict",
        headers={
            "Origin": "http://untrusted-site.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert blocked.headers.get("access-control-allow-origin") is None
