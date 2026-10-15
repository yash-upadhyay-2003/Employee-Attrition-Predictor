"""
Workforce Intelligence — Employee Attrition Risk Estimator (Frontend)
====================================================================
This module implements the decoupled Streamlit frontend for the Employee Attrition
Predictor application. It collects 30 raw employee attributes through a structured
enterprise web interface, validates input consistency, and communicates with a remote
or local FastAPI backend via HTTP REST endpoints.

Architecture & Design Principles:
---------------------------------
1. Zero Local Inference:
   - This frontend contains NO machine learning models, NO joblib deserializers,
     and NO local Scikit-Learn pipelines.
   - All feature preprocessing, ratio engineering, and model inference are delegated
     entirely to the FastAPI prediction service (`POST /predict`).

2. Resilient HTTP Transport:
   - Communicates with the backend using the `requests` library.
   - Configurable cold-start timeouts: Free-tier cloud instances (e.g., Render)
     often spin down when idle. The frontend uses a 5-second connect timeout and a
     90-second read timeout accompanied by an informative progress spinner.

3. Safe URL Configuration:
   - Resolves the backend base URL dynamically from the `BACKEND_API_URL` environment
     variable or Streamlit Secrets, with a safe fallback to `http://127.0.0.1:8000`.
   - Normalizes schemes and trailing slashes safely.

4. Responsible Governance:
   - Retains all ethical AI disclaimers, neutral risk classifications, and non-prescriptive
     guidelines to ensure workforce evaluations remain human-centered.
"""

import os
import re
from typing import Dict, Any, Tuple, Optional, List
import requests
import streamlit as st


# =============================================================================
# 1. Backend URL Resolution & Normalization Helper
# =============================================================================

def get_backend_url(custom_url: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve and validate the backend API base URL.

    Resolution Priority:
    --------------------
    1. Explicit custom_url parameter (useful for testing or direct override).
    2. Streamlit Cloud Secrets (`st.secrets["BACKEND_API_URL"]`).
    3. Operating system environment variable (`os.getenv("BACKEND_API_URL")`).
    4. Local development default (`http://127.0.0.1:8000`).

    Validation & Normalization Rules:
    ---------------------------------
    - Must start with a valid HTTP or HTTPS scheme (e.g., 'http://' or 'https://').
    - Trailing slashes are stripped to prevent double-slash path concatenation (e.g., 'url//predict').
    - Empty or malformed strings return a clear error description.

    Returns:
    --------
    Tuple[Optional[str], Optional[str]]:
        (normalized_url, None) on success.
        (None, error_message) if the configured URL is invalid.
    """
    raw_url: Optional[str] = custom_url

    # Check Streamlit Cloud Secrets if no direct override is given
    if not raw_url:
        try:
            if hasattr(st, "secrets") and "BACKEND_API_URL" in st.secrets:
                raw_url = str(st.secrets["BACKEND_API_URL"])
        except Exception:
            # Streamlit secrets file may not exist in local mode; ignore cleanly
            pass

    # Check environment variable if still unresolved
    if not raw_url:
        raw_url = os.getenv("BACKEND_API_URL")

    # Fallback to local FastAPI development server default
    if not raw_url:
        raw_url = "http://127.0.0.1:8000"

    raw_url = raw_url.strip()

    # Verify presence of valid HTTP/HTTPS scheme
    if not (raw_url.startswith("http://") or raw_url.startswith("https://")):
        return None, (
            f"Invalid BACKEND_API_URL '{raw_url}': URL must begin with 'http://' or 'https://'."
        )

    # Safely strip trailing slashes
    normalized_url = raw_url.rstrip("/")
    return normalized_url, None


# =============================================================================
# 2. Payload Construction Helper
# =============================================================================

def prepare_prediction_payload(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert UI form inputs into the exact 30-feature dictionary expected by
    the FastAPI backend `EmployeePredictionRequest` schema.

    Important Architecture Note:
    ----------------------------
    The frontend NEVER sends engineered tenure ratios (CompanyExperienceRatio,
    CurrentRoleTenureRatio, ManagerTenureRatio). The backend calculates them
    server-side with zero-denominator resilience.

    Parameters:
    -----------
    form_data : Dict[str, Any]
        Raw input values gathered from the Streamlit widgets.

    Returns:
    --------
    Dict[str, Any]:
        Strictly typed 30-feature dictionary using PascalCase aliases.
    """
    payload = {
        # 1. Demographics & Alignment (9 features)
        "Age": int(form_data["age"]),
        "Gender": str(form_data["gender"]),
        "MaritalStatus": str(form_data["marital_status"]),
        "Department": str(form_data["department"]),
        "JobRole": str(form_data["job_role"]),
        "JobLevel": int(form_data["job_level"]),
        "Education": int(form_data["education"]),
        "EducationField": str(form_data["education_field"]),
        "BusinessTravel": str(form_data["business_travel"]),

        # 2. Compensation & Pay Structure (6 features)
        "MonthlyIncome": float(form_data["monthly_income"]),
        "StockOptionLevel": int(form_data["stock_option_level"]),
        "DailyRate": int(form_data["daily_rate"]),
        "HourlyRate": int(form_data["hourly_rate"]),
        "MonthlyRate": int(form_data["monthly_rate"]),
        "PercentSalaryHike": int(form_data["percent_salary_hike"]),

        # 3. Career History & Tenure (6 features)
        "TotalWorkingYears": int(form_data["total_working_years"]),
        "YearsAtCompany": int(form_data["years_at_company"]),
        "YearsInCurrentRole": int(form_data["years_in_current_role"]),
        "YearsWithCurrManager": int(form_data["years_with_curr_manager"]),
        "YearsSinceLastPromotion": int(form_data["years_since_last_promotion"]),
        "NumCompaniesWorked": int(form_data["num_companies_worked"]),

        # 4. Workplace Environment & Satisfaction (9 features)
        "OverTime": str(form_data["over_time"]),
        "DistanceFromHome": int(form_data["distance_from_home"]),
        "TrainingTimesLastYear": int(form_data["training_times_last_year"]),
        "JobSatisfaction": int(form_data["job_satisfaction"]),
        "EnvironmentSatisfaction": int(form_data["environment_satisfaction"]),
        "RelationshipSatisfaction": int(form_data["relationship_satisfaction"]),
        "WorkLifeBalance": int(form_data["work_life_balance"]),
        "JobInvolvement": int(form_data["job_involvement"]),
        "PerformanceRating": int(form_data["performance_rating"]),
    }
    return payload


# =============================================================================
# 3. Response Validation Helper
# =============================================================================

REQUIRED_RESPONSE_KEYS = [
    "prediction",
    "predicted_label",
    "probability_attrition",
    "probability_retention",
    "is_elevated_risk",
    "status_text",
    "decision_threshold",
    "computed_ratios"
]

def validate_prediction_response(response_data: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate that the JSON object returned by the backend contains all mandatory
    fields matching the EmployeePredictionResponse contract.

    Returns:
    --------
    Tuple[bool, Optional[str]]:
        (True, None) if response conforms to schema.
        (False, error_detail) if incomplete or malformed.
    """
    if not isinstance(response_data, dict):
        return False, "Response payload is not a valid JSON dictionary."

    missing = [key for key in REQUIRED_RESPONSE_KEYS if key not in response_data]
    if missing:
        return False, f"Backend response is missing required fields: {', '.join(missing)}"

    ratios = response_data.get("computed_ratios")
    if not isinstance(ratios, dict):
        return False, "Response field 'computed_ratios' is missing or not a dictionary."

    ratio_keys = ["company_experience_ratio", "current_role_tenure_ratio", "manager_tenure_ratio"]
    missing_ratios = [k for k in ratio_keys if k not in ratios]
    if missing_ratios:
        return False, f"'computed_ratios' is missing required fields: {', '.join(missing_ratios)}"

    return True, None


# =============================================================================
# 4. HTTP Client Request Execution Helper
# =============================================================================

def call_prediction_api(
    payload: Dict[str, Any],
    backend_url: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 90.0,
    session: Optional[requests.Session] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
    """
    Dispatch an HTTP POST request to the FastAPI `/predict` endpoint.

    Resilience Considerations:
    --------------------------
    - Uses a short connection timeout (5s) to detect unreachable hosts quickly.
    - Uses a longer read timeout (90s) to accommodate Render free-tier cold starts,
      where idle instances can require 50-70 seconds to spin up.
    - Never logs or exposes employee input values.
    - Catches all transport exceptions and maps them to clear, actionable messages.

    Returns:
    --------
    Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
        (response_data, error_message, http_status_code)
    """
    endpoint = f"{backend_url}/predict"
    http_client = session if session is not None else requests

    try:
        response = http_client.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=(connect_timeout, read_timeout)
        )
    except requests.exceptions.ConnectTimeout:
        return None, (
            f"Connection timed out while connecting to the backend at '{backend_url}'. "
            "Please verify that the host is reachable and running."
        ), None
    except requests.exceptions.ReadTimeout:
        return None, (
            "The prediction service timed out while generating a response (exceeded 90 seconds). "
            "If the backend is hosted on a free cloud service (such as Render), it may be waking up "
            "from an idle sleep. Please wait 15-30 seconds and retry your request."
        ), None
    except requests.exceptions.ConnectionError:
        return None, (
            f"Unable to connect to the prediction backend at '{backend_url}'. "
            "Please ensure the FastAPI service is started and that BACKEND_API_URL is configured correctly. "
            "You can retry once the backend is online."
        ), None
    except requests.exceptions.RequestException as req_err:
        return None, (
            f"A network transport error occurred while communicating with the prediction backend: {type(req_err).__name__}"
        ), None

    status_code = response.status_code

    # Handle Successful Inference (HTTP 200)
    if status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return None, "Backend returned an HTTP 200 response, but the content was not valid JSON.", status_code

        valid, val_err = validate_prediction_response(data)
        if not valid:
            return None, f"Malformed response from prediction service: {val_err}", status_code

        return data, None, status_code

    # Handle Validation Rejection (HTTP 422)
    if status_code == 422:
        try:
            err_json = response.json()
            detail = err_json.get("detail", "Input validation rejected by server.")
            if isinstance(detail, list):
                # Format Pydantic validation error lists cleanly
                formatted_errors = [f"{e.get('loc', ['field'])[-1]}: {e.get('msg', 'invalid')}" for e in detail if isinstance(e, dict)]
                detail = "; ".join(formatted_errors) if formatted_errors else str(detail)
            return None, f"Input Validation Error (HTTP 422): {detail}", status_code
        except Exception:
            return None, "Input Validation Error (HTTP 422): The submitted features were rejected by the server schema.", status_code

    # Handle Service Degradation / Maintenance (HTTP 503)
    if status_code == 503:
        try:
            detail = response.json().get("detail", "Prediction service is temporarily unavailable.")
            return None, f"Service Unavailable (HTTP 503): {detail}", status_code
        except Exception:
            return None, "Service Unavailable (HTTP 503): The prediction engine is temporarily initializing or degraded.", status_code

    # Handle Server Internal Error (HTTP 500)
    if status_code == 500:
        return None, (
            "Internal Backend Error (HTTP 500): The prediction service encountered an unexpected internal error. "
            "Please check backend service logs or retry."
        ), status_code

    # Catch-all for other HTTP statuses
    return None, f"Prediction Service returned unexpected HTTP status {status_code}.", status_code


# =============================================================================
# 5. Streamlit Page Setup & Corporate CSS
# =============================================================================

def setup_page():
    """Configure Streamlit browser tab settings and corporate styling."""
    st.set_page_config(
        page_title="Workforce Intelligence — Employee Attrition Risk Estimator",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Enterprise CSS: Identical to root app.py design standards
    st.markdown("""
    <style>
        /* Global Background & Typography */
        .stApp {
            background-color: #FFFFFF;
            color: #111827;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Main Content Container */
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #111827 !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }

        /* Top Header Bar */
        .header-container {
            border-bottom: 1px solid #E5E7EB;
            padding-bottom: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .system-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background-color: #F3F4F6;
            color: #4B5563;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            border: 1px solid #E5E7EB;
            margin-bottom: 0.5rem;
        }

        /* Structured Panels */
        .panel-card {
            background-color: #FAFAFA;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }

        /* Ethical Advisory Notice */
        .advisory-box {
            background-color: #F9FAFB;
            border-left: 4px solid #EA580C;
            border-top: 1px solid #E5E7EB;
            border-right: 1px solid #E5E7EB;
            border-bottom: 1px solid #E5E7EB;
            padding: 1rem 1.25rem;
            border-radius: 0 4px 4px 0;
            margin-bottom: 1.75rem;
            font-size: 0.875rem;
            color: #374151;
            line-height: 1.5;
        }

        /* Results Card */
        .result-card-elevated {
            background-color: #FFFDFD;
            border: 1px solid #FCA5A5;
            border-left: 6px solid #DC2626;
            border-radius: 6px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }
        .result-card-retained {
            background-color: #FDFDFD;
            border: 1px solid #86EFAC;
            border-left: 6px solid #16A34A;
            border-radius: 6px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }

        /* Metric Badges */
        .metric-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.1;
        }
        .metric-subtext {
            font-size: 0.8125rem;
            color: #6B7280;
            margin-top: 0.25rem;
        }

        /* Form Inputs & Controls */
        .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #374151 !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }

        /* Primary Corporate Button */
        .stButton > button {
            background-color: #EA580C !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            border: none !important;
            padding: 0.65rem 1.75rem !important;
            font-size: 0.95rem !important;
            transition: background-color 0.15s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #C2410C !important;
            color: #FFFFFF !important;
        }
        .stButton > button:active {
            background-color: #9A3412 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# 6. Main Interactive Application
# =============================================================================

def main():
    """Main rendering entry point for the decoupled Streamlit frontend."""
    setup_page()

    # -------------------------------------------------------------------------
    # A. Header & Responsible Governance Advisory
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="header-container">
        <div class="system-badge">Enterprise Workforce Analytics Console — Web Client</div>
        <h1 style="margin: 0; font-size: 1.875rem;">Employee Attrition Risk Estimator</h1>
        <p style="margin: 0.35rem 0 0 0; color: #4B5563; font-size: 0.95rem;">
            Statistical decision-support tool powered by a validated Scikit-Learn classification pipeline hosted on FastAPI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ethical AI Notice (Preserved exactly from root app.py)
    st.markdown("""
    <div class="advisory-box">
        <strong>Ethical AI & Decision-Support Notice:</strong><br>
        Predictions generated by this model are probabilistic statistical indicators derived from historical workforce patterns. 
        They <strong>do not establish causality</strong> and must <strong>never</strong> be used as the sole or automated basis 
        for termination, compensation, promotion, or adverse employment actions. All outputs are intended strictly to support 
        human-led talent retention discussions and workload management.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # B. Backend Connection Verification
    # -------------------------------------------------------------------------
    backend_url, url_error = get_backend_url()
    if url_error:
        st.error(f"Configuration Error: {url_error}")
        st.info("Please set the 'BACKEND_API_URL' environment variable (e.g., 'http://127.0.0.1:8000').")
        st.stop()

    # -------------------------------------------------------------------------
    # C. Form Input Controls (Preserving All 30 Raw Controls & Layouts)
    # -------------------------------------------------------------------------
    st.markdown("### Employee Profile Data Collection")
    st.markdown(
        "<p style='color: #6B7280; font-size: 0.875rem; margin-top: -0.5rem; margin-bottom: 1.25rem;'>"
        "Enter the employee's demographic, departmental, compensation, and satisfaction attributes to evaluate turnover risk.</p>",
        unsafe_allow_html=True
    )

    with st.form(key="attrition_eval_form"):
        # Section 1: Demographics & Organizational Alignment
        with st.expander("1. Demographics & Departmental Alignment", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", min_value=18, max_value=65, value=35, step=1, help="Employee chronological age.")
                gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
                marital_status = st.selectbox("Marital Status", options=["Married", "Single", "Divorced"], index=0)
            with col2:
                department = st.selectbox("Department", options=["Research & Development", "Sales", "Human Resources"], index=0)
                job_role = st.selectbox("Job Role", options=[
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources"
                ], index=0)
                job_level = st.select_slider("Job Level", options=[1, 2, 3, 4, 5], value=2, help="Hierarchical job tier (1: Entry, 5: Executive).")
            with col3:
                education = st.selectbox("Education Level", options=[1, 2, 3, 4, 5], index=2, format_func=lambda x: {
                    1: "1 — Below College", 2: "2 — College", 3: "3 — Bachelor", 4: "4 — Master", 5: "5 — Doctor"
                }[x])
                education_field = st.selectbox("Education Field", options=[
                    "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
                ], index=0)
                business_travel = st.selectbox("Business Travel Frequency", options=["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=0)

        # Section 2: Compensation & Pay Rates
        with st.expander("2. Compensation & Pay Structure", expanded=True):
            col4, col5, col6 = st.columns(3)
            with col4:
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100, help="Monthly gross salary.")
                stock_option_level = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], index=1, help="Tier of stock grant (0: None, 3: Highest).")
            with col5:
                daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=10)
                hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65, step=1)
            with col6:
                monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=14000, step=100)
                percent_salary_hike = st.slider("Last Salary Hike (%)", min_value=10, max_value=25, value=15, step=1)

        # Section 3: Career History & Tenure Metrics
        with st.expander("3. Experience & Tenure Trajectory", expanded=True):
            col7, col8, col9 = st.columns(3)
            with col7:
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10, step=1, help="Total career work experience.")
                years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5, step=1, help="Tenure at current organization.")
            with col8:
                years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=25, value=3, step=1)
                years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=25, value=3, step=1)
            with col9:
                years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, step=1)
                num_companies_worked = st.number_input("Number of Prior Companies", min_value=0, max_value=10, value=2, step=1)

        # Section 4: Workplace Environment, Performance & Satisfaction
        with st.expander("4. Workplace Environment & Satisfaction Indicators", expanded=True):
            col10, col11, col12 = st.columns(3)
            with col10:
                over_time = st.selectbox("Overtime Status", options=["No", "Yes"], index=0, help="Whether the employee works contracted overtime hours.")
                distance_from_home = st.slider("Distance From Home (km)", min_value=1, max_value=30, value=7, step=1)
                training_times_last_year = st.slider("Training Sessions Attended Last Year", min_value=0, max_value=6, value=2, step=1)
            with col11:
                job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3, format_func=lambda x: {
                    1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
                }[x])
                environment_satisfaction = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3, format_func=lambda x: {
                    1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
                }[x])
                relationship_satisfaction = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3, format_func=lambda x: {
                    1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
                }[x])
            with col12:
                work_life_balance = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=3, format_func=lambda x: {
                    1: "1 — Bad", 2: "2 — Good", 3: "3 — Better", 4: "4 — Best"
                }[x])
                job_involvement = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3, format_func=lambda x: {
                    1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
                }[x])
                performance_rating = st.selectbox("Performance Rating", options=[3, 4], index=0, format_func=lambda x: {
                    3: "3 — Excellent", 4: "4 — Outstanding"
                }[x], help="Employee evaluation rating (3: Excellent, 4: Outstanding in IBM HR dataset).")

        # Form Submission Button
        submit_button = st.form_submit_button(label="Evaluate Attrition Risk Indicator")

    # -------------------------------------------------------------------------
    # D. Form Submission & HTTP API Communication
    # -------------------------------------------------------------------------
    if submit_button:
        # Pre-flight domain consistency check (Client-Side Guidance)
        client_warnings = []
        if years_at_company > total_working_years:
            client_warnings.append("Years at Company cannot exceed Total Working Years.")
        if years_in_current_role > years_at_company:
            client_warnings.append("Years in Current Role cannot exceed Years at Company.")
        if years_with_curr_manager > years_at_company:
            client_warnings.append("Years with Current Manager cannot exceed Years at Company.")
        if years_since_last_promotion > years_at_company:
            client_warnings.append("Years Since Last Promotion cannot exceed Years at Company.")

        if client_warnings:
            for warn in client_warnings:
                st.warning(f"Input Verification Note: {warn}")

        # Assemble the 30 raw features dictionary
        form_dict = {
            "age": age, "gender": gender, "marital_status": marital_status,
            "department": department, "job_role": job_role, "job_level": job_level,
            "education": education, "education_field": education_field, "business_travel": business_travel,
            "monthly_income": monthly_income, "stock_option_level": stock_option_level,
            "daily_rate": daily_rate, "hourly_rate": hourly_rate, "monthly_rate": monthly_rate,
            "percent_salary_hike": percent_salary_hike, "total_working_years": total_working_years,
            "years_at_company": years_at_company, "years_in_current_role": years_in_current_role,
            "years_with_curr_manager": years_with_curr_manager,
            "years_since_last_promotion": years_since_last_promotion,
            "num_companies_worked": num_companies_worked, "over_time": over_time,
            "distance_from_home": distance_from_home,
            "training_times_last_year": training_times_last_year,
            "job_satisfaction": job_satisfaction,
            "environment_satisfaction": environment_satisfaction,
            "relationship_satisfaction": relationship_satisfaction,
            "work_life_balance": work_life_balance, "job_involvement": job_involvement,
            "performance_rating": performance_rating
        }

        payload = prepare_prediction_payload(form_dict)

        # Dispatch HTTP request with cold-start progress spinner
        connect_timeout = float(os.getenv("API_CONNECT_TIMEOUT", 5.0))
        read_timeout = float(os.getenv("API_READ_TIMEOUT", 90.0))

        spinner_msg = (
            "Connecting to prediction service... "
            "(If the service is waking up from cold sleep, this may take up to 60-90 seconds)"
        )
        with st.spinner(spinner_msg):
            api_data, api_error, status_code = call_prediction_api(
                payload=payload,
                backend_url=backend_url,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout
            )

        # Handle API Errors Gracefully (Never fall back silently to local inference)
        if api_error:
            st.error(f"Prediction Service Unavailable: {api_error}")
            st.info("The application could not complete inference. Please verify backend status and try again.")
            return

        if not api_data:
            st.error("Prediction Error: No response received from the prediction backend.")
            return

        # ---------------------------------------------------------------------
        # E. Render Diagnostic Assessment Card
        # ---------------------------------------------------------------------
        prob_attrition = float(api_data["probability_attrition"])
        is_elevated_risk = bool(api_data["is_elevated_risk"])
        computed_ratios = api_data.get("computed_ratios", {})
        backend_warnings = api_data.get("warnings", [])

        st.markdown("---")
        st.markdown("### Risk Diagnostic Assessment")

        if is_elevated_risk:
            st.markdown(f"""
            <div class="result-card-elevated">
                <div style="font-size: 0.8125rem; font-weight: 700; color: #DC2626; text-transform: uppercase; letter-spacing: 0.05em;">
                    Status: Elevated Model-Estimated Attrition Probability
                </div>
                <div style="display: flex; align-items: baseline; gap: 1.5rem; margin-top: 0.5rem;">
                    <div>
                        <div class="metric-value" style="color: #DC2626;">{prob_attrition * 100:.1f}%</div>
                        <div class="metric-subtext">Estimated Probability of Departure (Statistical Estimate)</div>
                    </div>
                    <div style="border-left: 1px solid #E5E7EB; padding-left: 1.5rem;">
                        <div style="font-size: 1.25rem; font-weight: 600; color: #374151;">Model Decision Classification: Attrition = Yes (1)</div>
                        <div class="metric-subtext">Standard Decision Boundary: 50.0%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-retained">
                <div style="font-size: 0.8125rem; font-weight: 700; color: #16A34A; text-transform: uppercase; letter-spacing: 0.05em;">
                    Status: Lower Model-Estimated Attrition Probability
                </div>
                <div style="display: flex; align-items: baseline; gap: 1.5rem; margin-top: 0.5rem;">
                    <div>
                        <div class="metric-value" style="color: #16A34A;">{prob_attrition * 100:.1f}%</div>
                        <div class="metric-subtext">Estimated Probability of Departure (Statistical Estimate)</div>
                    </div>
                    <div style="border-left: 1px solid #E5E7EB; padding-left: 1.5rem;">
                        <div style="font-size: 1.25rem; font-weight: 600; color: #374151;">Model Decision Classification: Attrition = No (0)</div>
                        <div class="metric-subtext">Standard Decision Boundary: 50.0%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Statistical Interpretation Note (Identical to root app.py)
        st.markdown("""
        <div style="margin-top: 1rem; padding: 0.85rem 1rem; background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 4px; font-size: 0.8125rem; color: #4B5563; line-height: 1.45;">
            <strong>Statistical Interpretation Note:</strong> This percentage represents an algorithmic estimate derived from historical patterns in the IBM HR Analytics dataset. It is a decision-support indicator, not a guaranteed outcome or a definitively calibrated individual probability. It must not be interpreted as certainty that an employee will stay or leave.
        </div>
        """, unsafe_allow_html=True)

        # Contextual Metric Cards (Computed server-side)
        st.markdown("<h4 style='margin-top: 1.5rem;'>Computed Trajectory Ratios & Context</h4>", unsafe_allow_html=True)
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        comp_exp = float(computed_ratios.get("company_experience_ratio", 0.0))
        role_stab = float(computed_ratios.get("current_role_tenure_ratio", 0.0))
        mgr_align = float(computed_ratios.get("manager_tenure_ratio", 0.0))

        with rcol1:
            st.metric("Company Tenure Ratio", f"{comp_exp:.2f}", help="YearsAtCompany / TotalWorkingYears (computed by backend)")
        with rcol2:
            st.metric("Role Stability Ratio", f"{role_stab:.2f}", help="YearsInCurrentRole / YearsAtCompany (computed by backend)")
        with rcol3:
            st.metric("Manager Alignment Ratio", f"{mgr_align:.2f}", help="YearsWithCurrManager / YearsAtCompany (computed by backend)")
        with rcol4:
            st.metric("Overtime Impact Flag", str(over_time), help="High correlation factor identified in EDA")

        # Display server-side advisory warnings if present
        if backend_warnings:
            for b_warn in backend_warnings:
                st.warning(f"Backend Advisory: {b_warn}")

        # Responsible HR Review Guidelines (Identical to root app.py)
        st.markdown("""
        <div style="background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 1rem 1.25rem; margin-top: 1.25rem; font-size: 0.875rem; color: #4B5563;">
            <strong style="color: #111827;">Responsible Human Review & Governance Guidelines:</strong>
            <ul style="margin: 0.5rem 0 0 1.25rem; padding: 0;">
                <li><strong>Exploratory Context:</strong> Use this indicator solely as an exploratory diagnostic signal to inform broad workforce discussions, not as a conclusive employee evaluation.</li>
                <li><strong>Human-in-the-Loop:</strong> Any follow-up must be supportive, voluntary, and conducted through standard two-way management check-ins.</li>
                <li><strong>Correlation vs. Causation:</strong> Predictive factors (e.g., tenure ratios, overtime status) are historical statistical correlations, not proven causal drivers for any specific individual.</li>
                <li><strong>Strict Policy Prohibition:</strong> This statistical estimate must not be entered into official personnel records or used as the sole basis for hiring, promotion, performance assessment, compensation, or termination decisions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # F. Technical Footer
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8125rem; color: #9CA3AF;">
        <div><strong>Model Pipeline:</strong> Tuned Logistic Regression (Scikit-Learn 1.6.1) | 80:20 Stratified Partition</div>
        <div><strong>Backend API:</strong> Connected to <code>{backend_url}</code></div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
