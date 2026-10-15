"""
Unit & Integration Tests for Streamlit Frontend HTTP Client
============================================================
This test module verifies the frontend HTTP transport layer, request formatting,
cold-start timeout handling, error resilience, and strict separation from model
inference artifacts.

All tests utilize mocked HTTP requests without requiring real network access
or a running backend process.
"""

import os
import ast
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests
import pytest

from frontend.app import (
    get_backend_url,
    prepare_prediction_payload,
    validate_prediction_response,
    call_prediction_api,
    REQUIRED_RESPONSE_KEYS
)
from backend.schemas import EmployeePredictionRequest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_form_inputs():
    """Provides a realistic set of 30 form input values matching root app.py defaults."""
    return {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "department": "Research & Development",
        "job_role": "Sales Executive",
        "job_level": 2,
        "education": 3,
        "education_field": "Life Sciences",
        "business_travel": "Travel_Rarely",
        "monthly_income": 5000.0,
        "stock_option_level": 1,
        "daily_rate": 800,
        "hourly_rate": 65,
        "monthly_rate": 14000,
        "percent_salary_hike": 15,
        "total_working_years": 10,
        "years_at_company": 5,
        "years_in_current_role": 3,
        "years_with_curr_manager": 3,
        "years_since_last_promotion": 1,
        "num_companies_worked": 2,
        "over_time": "No",
        "distance_from_home": 7,
        "training_times_last_year": 2,
        "job_satisfaction": 3,
        "environment_satisfaction": 3,
        "relationship_satisfaction": 3,
        "work_life_balance": 3,
        "job_involvement": 3,
        "performance_rating": 3,
    }


@pytest.fixture
def sample_valid_backend_response():
    """Provides a mock response conforming strictly to EmployeePredictionResponse."""
    return {
        "prediction": 0,
        "predicted_label": "No",
        "probability_attrition": 0.185,
        "probability_retention": 0.815,
        "is_elevated_risk": False,
        "status_text": "LOWER MODEL-ESTIMATED ATTRITION PROBABILITY",
        "decision_threshold": 0.50,
        "computed_ratios": {
            "company_experience_ratio": 0.50,
            "current_role_tenure_ratio": 0.60,
            "manager_tenure_ratio": 0.60
        },
        "warnings": []
    }


# -----------------------------------------------------------------------------
# 1. URL Resolution & Normalization Tests
# -----------------------------------------------------------------------------

def test_01_backend_url_resolution_and_normalization():
    """
    Test that get_backend_url correctly resolves default, strips trailing slashes,
    validates schemes, and handles environment overrides safely.
    """
    # 1. Default fallback when no env var is set
    with patch.dict(os.environ, {}, clear=True):
        url, err = get_backend_url()
        assert err is None
        assert url == "http://127.0.0.1:8000"

    # 2. Strips trailing slashes safely
    url, err = get_backend_url(custom_url="http://127.0.0.1:8000/")
    assert err is None
    assert url == "http://127.0.0.1:8000"

    url, err = get_backend_url(custom_url="https://attrition-api.onrender.com///")
    assert err is None
    assert url == "https://attrition-api.onrender.com"

    # 3. Environment variable configuration
    with patch.dict(os.environ, {"BACKEND_API_URL": "https://api.workforce.enterprise.com"}):
        url, err = get_backend_url()
        assert err is None
        assert url == "https://api.workforce.enterprise.com"

    # 4. Reject URLs without valid scheme
    url, err = get_backend_url(custom_url="localhost:8000")
    assert url is None
    assert "must begin with 'http://' or 'https://'" in err

    url, err = get_backend_url(custom_url="ftp://api.internal/predict")
    assert url is None
    assert "must begin with 'http://' or 'https://'" in err


# -----------------------------------------------------------------------------
# 2. Payload Completeness & Ratio Exclusion Tests
# -----------------------------------------------------------------------------

def test_02_payload_structure_contains_all_30_raw_inputs(sample_form_inputs):
    """
    Verify that prepare_prediction_payload converts the 30 UI inputs into the
    exact schema expected by the backend and STRICTLY EXCLUDES engineered ratios.
    """
    payload = prepare_prediction_payload(sample_form_inputs)

    # Must contain exactly 30 raw fields
    assert len(payload) == 30, f"Expected exactly 30 fields, got {len(payload)}"

    # Engineered ratios MUST NOT be sent in the client payload
    forbidden_ratios = [
        "CompanyExperienceRatio",
        "CurrentRoleTenureRatio",
        "ManagerTenureRatio",
        "company_experience_ratio",
        "current_role_tenure_ratio",
        "manager_tenure_ratio"
    ]
    for ratio in forbidden_ratios:
        assert ratio not in payload, f"Engineered ratio '{ratio}' must not be sent by frontend."

    # Validate raw fields existence and types
    assert payload["Age"] == 35
    assert isinstance(payload["Age"], int)
    assert payload["MonthlyIncome"] == 5000.0
    assert isinstance(payload["MonthlyIncome"], float)
    assert payload["Gender"] == "Male"
    assert payload["OverTime"] == "No"
    assert payload["PerformanceRating"] == 3


def test_03_payload_schema_compatibility_with_backend_pydantic_model(sample_form_inputs):
    """
    Verify that the payload produced by the frontend successfully validates against
    the backend Pydantic EmployeePredictionRequest schema without validation errors.
    """
    payload = prepare_prediction_payload(sample_form_inputs)
    # Instantiate Pydantic model directly to prove 100% schema alignment
    req = EmployeePredictionRequest(**payload)
    assert req.age == 35
    assert req.gender == "Male"
    assert req.monthly_income == 5000.0


# -----------------------------------------------------------------------------
# 3. HTTP Request & Response Validation Tests
# -----------------------------------------------------------------------------

def test_04_successful_http_prediction_request(sample_form_inputs, sample_valid_backend_response):
    """
    Verify successful end-to-end HTTP request handling on HTTP 200 response.
    """
    payload = prepare_prediction_payload(sample_form_inputs)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_valid_backend_response

    with patch("requests.post", return_value=mock_response) as mock_post:
        data, err, code = call_prediction_api(
            payload=payload,
            backend_url="http://127.0.0.1:8000",
            connect_timeout=5.0,
            read_timeout=90.0
        )

        assert err is None
        assert code == 200
        assert data is not None
        assert data["prediction"] == 0
        assert data["probability_attrition"] == 0.185
        assert data["is_elevated_risk"] is False

        # Verify call arguments
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/predict",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=(5.0, 90.0)
        )


def test_05_read_timeout_handling_render_cold_start(sample_form_inputs):
    """
    Verify graceful handling of HTTP read timeouts (e.g. Render 50-70s cold start).
    """
    payload = prepare_prediction_payload(sample_form_inputs)

    with patch("requests.post", side_effect=requests.exceptions.ReadTimeout("Read timed out")):
        data, err, code = call_prediction_api(
            payload=payload,
            backend_url="http://127.0.0.1:8000"
        )

        assert data is None
        assert code is None
        assert err is not None
        assert "timed out while generating a response" in err
        assert "Render" in err or "cloud" in err
        assert "retry" in err.lower()


def test_06_connection_error_handling(sample_form_inputs):
    """
    Verify graceful handling of network connection failures when backend is offline.
    """
    payload = prepare_prediction_payload(sample_form_inputs)

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        data, err, code = call_prediction_api(
            payload=payload,
            backend_url="http://127.0.0.1:8000"
        )

        assert data is None
        assert code is None
        assert err is not None
        assert "Unable to connect to the prediction backend" in err
        assert "http://127.0.0.1:8000" in err


def test_07_http_422_validation_error_handling(sample_form_inputs):
    """
    Verify clear user messaging when the backend rejects payload with HTTP 422.
    """
    payload = prepare_prediction_payload(sample_form_inputs)
    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.json.return_value = {
        "detail": [
            {"loc": ["body", "Age"], "msg": "Input should be greater than or equal to 18", "type": "greater_than_equal"}
        ]
    }

    with patch("requests.post", return_value=mock_response):
        data, err, code = call_prediction_api(
            payload=payload,
            backend_url="http://127.0.0.1:8000"
        )

        assert data is None
        assert code == 422
        assert err is not None
        assert "Input Validation Error (HTTP 422)" in err
        assert "Age: Input should be greater than or equal to 18" in err


def test_08_http_500_and_503_error_handling(sample_form_inputs):
    """
    Verify clear user messaging for internal server errors (500) and service unavailable (503).
    """
    payload = prepare_prediction_payload(sample_form_inputs)

    # Test HTTP 500
    mock_500 = MagicMock()
    mock_500.status_code = 500
    with patch("requests.post", return_value=mock_500):
        data, err, code = call_prediction_api(payload, "http://127.0.0.1:8000")
        assert data is None
        assert code == 500
        assert "Internal Backend Error (HTTP 500)" in err

    # Test HTTP 503
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.json.return_value = {"detail": "Model artifact is not loaded or failed initialization."}
    with patch("requests.post", return_value=mock_503):
        data, err, code = call_prediction_api(payload, "http://127.0.0.1:8000")
        assert data is None
        assert code == 503
        assert "Service Unavailable (HTTP 503)" in err
        assert "Model artifact is not loaded" in err


def test_09_invalid_or_incomplete_json_response():
    """
    Verify detection of non-JSON responses and responses with missing schema keys.
    """
    # 1. Non-JSON payload
    mock_bad_json = MagicMock()
    mock_bad_json.status_code = 200
    mock_bad_json.json.side_effect = ValueError("Invalid JSON format")

    with patch("requests.post", return_value=mock_bad_json):
        data, err, code = call_prediction_api({}, "http://127.0.0.1:8000")
        assert data is None
        assert code == 200
        assert "content was not valid JSON" in err

    # 2. Incomplete JSON missing required keys
    incomplete_data = {
        "prediction": 0,
        "probability_attrition": 0.20
        # Missing probability_retention, status_text, computed_ratios, etc.
    }
    valid, val_err = validate_prediction_response(incomplete_data)
    assert not valid
    assert "missing required fields" in val_err


# -----------------------------------------------------------------------------
# 4. Strict Zero-Local-Inference Audit Test
# -----------------------------------------------------------------------------

def test_10_no_model_loading_or_inference_imports_in_frontend():
    """
    Verify that frontend/app.py DOES NOT import joblib, scikit-learn,
    backend model_service, or load .joblib files on disk.
    """
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    assert frontend_path.exists(), f"File {frontend_path} does not exist"

    code_text = frontend_path.read_text(encoding="utf-8")
    parsed_ast = ast.parse(code_text)

    # Check all import statements
    imported_modules = set()
    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    forbidden_modules = {"joblib", "sklearn", "scipy", "model_service"}
    intersection = imported_modules.intersection(forbidden_modules)
    assert not intersection, f"frontend/app.py must not import: {intersection}"

    # Verify that .joblib files are not referenced for local loading
    assert "joblib.load" not in code_text, "frontend/app.py must not call joblib.load"
    assert "employee_attrition_model" not in code_text, (
        "frontend/app.py must not reference model artifact files directly"
    )
