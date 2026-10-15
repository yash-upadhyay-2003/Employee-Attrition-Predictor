"""
Automated Test Suite for FastAPI Prediction Backend
===================================================
This module validates the FastAPI backend using Starlette/FastAPI's TestClient.
It verifies health probes, model metadata, input validation schemas,
zero-denominator resilience, and exact prediction parity with direct Scikit-Learn
pipeline inference.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.model_service import model_service

client = TestClient(app)
ROOT_DIR = Path(__file__).resolve().parent.parent
ACTIVE_MODEL_PATH = ROOT_DIR / "employee_attrition_model_80_20.joblib"


@pytest.fixture
def base_valid_payload():
    """Returns a realistic, valid 30-feature employee dictionary matching app.py defaults."""
    return {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "department": "Research & Development",
        "job_role": "Research Scientist",
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
        "performance_rating": 3
    }


# =============================================================================
# Test 1: GET /health returns operational readiness
# =============================================================================
def test_01_health_endpoint():
    """Verify that /health reports healthy status and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data
    assert "environment" in data


# =============================================================================
# Test 2: GET /model-info returns verified metadata
# =============================================================================
def test_02_model_info_endpoint():
    """Verify that /model-info exposes verified architecture and validation metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Logistic Regression (Tuned)"
    assert data["artifact_filename"] == "employee_attrition_model_80_20.joblib"
    assert data["total_features"] == 33
    assert data["numerical_features_count"] == 26
    assert data["categorical_features_count"] == 7
    assert data["classes"] == [0, 1]
    assert data["decision_threshold"] == 0.50
    assert data["validation_cv_f1"] == 0.5913
    assert data["validation_cv_roc_auc"] == 0.844


# =============================================================================
# Test 3: POST /predict returns valid structural response for valid input
# =============================================================================
def test_03_predict_valid_request_structure(base_valid_payload):
    """Verify that a valid payload returns all expected fields with correct types."""
    response = client.post("/predict", json=base_valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert data["predicted_label"] in ["No", "Yes"]
    assert "probability_attrition" in data
    assert "probability_retention" in data
    assert "is_elevated_risk" in data
    assert "status_text" in data
    assert data["decision_threshold"] == 0.50
    assert "computed_ratios" in data
    assert "warnings" in data


# =============================================================================
# Test 4: Probabilities are bounded in [0, 1] and sum to approximately 1.0
# =============================================================================
def test_04_probabilities_range_and_sum(base_valid_payload):
    """Verify probability validity and mathematical distribution constraints."""
    response = client.post("/predict", json=base_valid_payload)
    assert response.status_code == 200
    data = response.json()
    p_att = data["probability_attrition"]
    p_ret = data["probability_retention"]
    
    assert 0.0 <= p_att <= 1.0
    assert 0.0 <= p_ret <= 1.0
    assert np.isclose(p_att + p_ret, 1.0, atol=1e-3)


# =============================================================================
# Test 5: Positive class probability extraction logic
# =============================================================================
def test_05_class_1_mapping_consistency(base_valid_payload):
    """Verify that probability_attrition directly reflects class 1 probability."""
    response = client.post("/predict", json=base_valid_payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check decision boundary consistency
    if data["probability_attrition"] >= 0.50:
        assert data["is_elevated_risk"] is True
        assert data["status_text"] == "Elevated Model-Estimated Attrition Probability"
        assert data["prediction"] == 1
        assert data["predicted_label"] == "Yes"
    else:
        assert data["is_elevated_risk"] is False
        assert data["status_text"] == "Lower Model-Estimated Attrition Probability"
        assert data["prediction"] == 0
        assert data["predicted_label"] == "No"


# =============================================================================
# Test 6: Invalid categorical values are rejected with 422
# =============================================================================
def test_06_invalid_categorical_rejected(base_valid_payload):
    """Verify that unapproved categorical options trigger HTTP 422 validation error."""
    bad_payload = base_valid_payload.copy()
    bad_payload["department"] = "Non_Existent_Department_XYZ"
    
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "department" in response.text


# =============================================================================
# Test 7: Missing required fields are rejected with 422
# =============================================================================
def test_07_missing_required_field_rejected(base_valid_payload):
    """Verify that omitting required features triggers HTTP 422 validation error."""
    bad_payload = base_valid_payload.copy()
    del bad_payload["monthly_income"]
    
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "monthlyincome" in response.text.lower()


# =============================================================================
# Test 8: Invalid numerical values are rejected according to constraints
# =============================================================================
def test_08_invalid_numeric_values_rejected(base_valid_payload):
    """Verify that out-of-range numerical values trigger HTTP 422 validation error."""
    # Test age under minimum 18
    bad_payload_age = base_valid_payload.copy()
    bad_payload_age["age"] = 10
    res_age = client.post("/predict", json=bad_payload_age)
    assert res_age.status_code == 422

    # Test monthly income under minimum 1000
    bad_payload_inc = base_valid_payload.copy()
    bad_payload_inc["monthly_income"] = 500.0
    res_inc = client.post("/predict", json=bad_payload_inc)
    assert res_inc.status_code == 422


# =============================================================================
# Test 9: Zero denominators produce exact zero-safe ratios (0.0)
# =============================================================================
def test_09_zero_denominator_ratio_resilience(base_valid_payload):
    """Verify that TotalWorkingYears=0 and YearsAtCompany=0 produce 0.0 without crash."""
    zero_tenure_payload = base_valid_payload.copy()
    zero_tenure_payload["total_working_years"] = 0
    zero_tenure_payload["years_at_company"] = 0
    zero_tenure_payload["years_in_current_role"] = 0
    zero_tenure_payload["years_with_curr_manager"] = 0
    zero_tenure_payload["years_since_last_promotion"] = 0

    response = client.post("/predict", json=zero_tenure_payload)
    assert response.status_code == 200
    data = response.json()
    ratios = data["computed_ratios"]
    assert ratios["company_experience_ratio"] == 0.0
    assert ratios["current_role_tenure_ratio"] == 0.0
    assert ratios["manager_tenure_ratio"] == 0.0


# =============================================================================
# Test 10: API prediction and probabilities match direct pipeline inference
# =============================================================================
def test_10_prediction_and_probability_parity_with_direct_pipeline(base_valid_payload):
    """
    Direct Parity Assertion:
    Compare API responses against direct in-memory inference through the original
    model pipeline for multiple distinct employee profiles.
    """
    direct_model = joblib.load(ACTIVE_MODEL_PATH)
    pos_idx = list(direct_model.classes_).index(1)

    # Test Case A: Standard baseline employee profile
    # Test Case B: High-risk employee profile (Sales Rep, OverTime=Yes, low salary, low tenure)
    # Test Case C: Zero-tenure new hire
    profiles = [
        base_valid_payload,
        {
            **base_valid_payload,
            "job_role": "Sales Representative",
            "over_time": "Yes",
            "monthly_income": 2000.0,
            "job_satisfaction": 1,
            "environment_satisfaction": 1,
            "years_at_company": 1,
            "total_working_years": 2,
            "years_in_current_role": 1,
            "years_with_curr_manager": 1
        },
        {
            **base_valid_payload,
            "total_working_years": 0,
            "years_at_company": 0,
            "years_in_current_role": 0,
            "years_with_curr_manager": 0,
            "years_since_last_promotion": 0
        }
    ]

    for idx, p in enumerate(profiles):
        # 1. API Call
        api_res = client.post("/predict", json=p)
        assert api_res.status_code == 200
        api_data = api_res.json()

        # 2. Direct Pipeline Computation
        # Compute ratios exactly
        twy = p["total_working_years"]
        yac = p["years_at_company"]
        c_ratio = float(yac) / float(twy) if twy > 0 else 0.0
        r_ratio = float(p["years_in_current_role"]) / float(yac) if yac > 0 else 0.0
        m_ratio = float(p["years_with_curr_manager"]) / float(yac) if yac > 0 else 0.0

        direct_df = pd.DataFrame({
            "Age": [int(p["age"])],
            "DailyRate": [int(p["daily_rate"])],
            "DistanceFromHome": [int(p["distance_from_home"])],
            "Education": [int(p["education"])],
            "EnvironmentSatisfaction": [int(p["environment_satisfaction"])],
            "HourlyRate": [int(p["hourly_rate"])],
            "JobInvolvement": [int(p["job_involvement"])],
            "JobLevel": [int(p["job_level"])],
            "JobSatisfaction": [int(p["job_satisfaction"])],
            "MonthlyIncome": [float(p["monthly_income"])],
            "MonthlyRate": [int(p["monthly_rate"])],
            "NumCompaniesWorked": [int(p["num_companies_worked"])],
            "PercentSalaryHike": [int(p["percent_salary_hike"])],
            "PerformanceRating": [int(p["performance_rating"])],
            "RelationshipSatisfaction": [int(p["relationship_satisfaction"])],
            "StockOptionLevel": [int(p["stock_option_level"])],
            "TotalWorkingYears": [int(p["total_working_years"])],
            "TrainingTimesLastYear": [int(p["training_times_last_year"])],
            "WorkLifeBalance": [int(p["work_life_balance"])],
            "YearsAtCompany": [int(p["years_at_company"])],
            "YearsInCurrentRole": [int(p["years_in_current_role"])],
            "YearsSinceLastPromotion": [int(p["years_since_last_promotion"])],
            "YearsWithCurrManager": [int(p["years_with_curr_manager"])],
            "CompanyExperienceRatio": [float(c_ratio)],
            "CurrentRoleTenureRatio": [float(r_ratio)],
            "ManagerTenureRatio": [float(m_ratio)],
            "BusinessTravel": [str(p["business_travel"])],
            "Department": [str(p["department"])],
            "EducationField": [str(p["education_field"])],
            "Gender": [str(p["gender"])],
            "JobRole": [str(p["job_role"])],
            "MaritalStatus": [str(p["marital_status"])],
            "OverTime": [str(p["over_time"])]
        })

        direct_pred = int(direct_model.predict(direct_df)[0])
        direct_prob = float(direct_model.predict_proba(direct_df)[0, pos_idx])

        # Assert bitwise prediction parity
        assert api_data["prediction"] == direct_pred, f"Profile {idx} prediction mismatch!"
        # Assert probability parity within 4 decimal places
        assert np.isclose(api_data["probability_attrition"], round(direct_prob, 4), atol=1e-4), \
            f"Profile {idx} probability mismatch! API: {api_data['probability_attrition']} vs Direct: {direct_prob}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
