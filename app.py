"""
Workforce Intelligence — Employee Attrition Risk Estimator
===========================================================
This module implements an interactive enterprise web application using Streamlit
to evaluate employee attrition risk based on historical organizational data.

End-to-End Application Flow:
----------------------------
1. Initialization:
   - Sets up page configuration (title, layout, enterprise CSS styling).
   - Defensively loads the trained Scikit-Learn pipeline from disk (cached in memory).
2. User Input Collection:
   - Presents an organized Streamlit form with 30 raw employee attributes
     (23 numerical, 7 categorical) matching the IBM HR Analytics schema.
3. Feature Engineering & Validation:
   - Validates logical consistency across user inputs (e.g., tenure constraints).
   - Computes 3 tenure ratios with zero-division resilience to produce 33 total model features.
4. Model Preprocessing & Inference:
   - Bundles inputs into a 1-row Pandas DataFrame with exact column names.
   - Passes data into the loaded Scikit-Learn ColumnTransformer (imputation + scaling + encoding).
   - Computes predictions and probabilities using the tuned Logistic Regression classifier.
   - Dynamically identifies the positive class ('Attrition = Yes' / 1) from pipeline.classes_.
5. Results Presentation & Responsible Governance:
   - Renders a color-coded diagnostic card (Elevated vs. Lower probability).
   - Displays computed tenure metrics and strict ethical non-prescriptive disclaimers.
"""

# Standard library modules
import os
from pathlib import Path  # Cross-platform, object-oriented filesystem paths (handles Windows vs Linux slashes)

# Third-party machine learning & data manipulation libraries
import joblib              # Efficient serialization/deserialization for scikit-learn models and numpy arrays
import numpy as np          # High-performance numerical computations and type conversions
import pandas as pd         # Tabular DataFrame creation, preserving column names required by ColumnTransformer
import streamlit as st      # Interactive web framework that executes Python scripts as reactive web applications

# -----------------------------------------------------------------------------
# 1. Page Configuration & Enterprise Styling
# -----------------------------------------------------------------------------
# st.set_page_config MUST be the first Streamlit command invoked in the script.
# It sets the browser tab title, favicon, and selects a wide layout so multi-column
# dashboards render cleanly on modern desktop screens without horizontal crowding.
st.set_page_config(
    page_title="Workforce Intelligence — Employee Attrition Risk Estimator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS: White background, slate typography, light borders, restrained orange accent
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

# -----------------------------------------------------------------------------
# 2. Model Loading (Cached & Defensive)
# -----------------------------------------------------------------------------
# Resolve the exact location of the trained model relative to this file's directory.
# Using Path(__file__).resolve().parent ensures the file is reliably found even if
# the user starts the Streamlit server from a different working directory.
MODEL_PATH = Path(__file__).resolve().parent / "employee_attrition_model_80_20.joblib"

@st.cache_resource(show_spinner=False)
def load_trained_pipeline():
    """
    Load the trained Scikit-Learn pipeline from disk with defensive error handling.

    Why this function exists:
    -------------------------
    Streamlit re-executes the entire Python script from line 1 on every user
    interaction (e.g., clicking a button or moving a slider). Without caching,
    the application would reload and unpickle the 8.7 KB model artifact from
    disk on every click. 

    The `@st.cache_resource` decorator caches the loaded Python object in memory
    across reruns and across multiple user sessions, ensuring instantaneous response.

    Returns:
    --------
    tuple:
        (pipeline, None) if the artifact exists and unpickles cleanly.
        (None, error_str) if the file is missing or unpickling fails.
    """
    # 1. Defensive Check: Confirm the model file exists on disk
    if not MODEL_PATH.exists():
        return None, f"Model artifact '{MODEL_PATH.name}' was not found in the application directory."
    
    # 2. Deserialization: Load the Scikit-Learn Pipeline object using joblib
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline, None
    except Exception as e:
        # Catch corrupted files, incompatible pickle formats, or dependency mismatches
        return None, f"Failed to deserialize model artifact: {str(e)}"

# -----------------------------------------------------------------------------
# 3. Header & Responsible Governance Advisory
# -----------------------------------------------------------------------------
# Displays enterprise branding, application title, and technical subtitle
st.markdown("""
<div class="header-container">
    <div class="system-badge">Enterprise Workforce Analytics Console</div>
    <h1 style="margin: 0; font-size: 1.875rem;">Employee Attrition Risk Estimator</h1>
    <p style="margin: 0.35rem 0 0 0; color: #4B5563; font-size: 0.95rem;">
        Statistical decision-support tool powered by a validated Scikit-Learn classification pipeline trained on an 80:20 stratified split.
    </p>
</div>
""", unsafe_allow_html=True)

# Ethical AI & Responsible Use Disclaimer:
# Machine learning in Human Resources carries legal and ethical considerations.
# Predictions must be framed as statistical indicators rather than definitive truths,
# preventing discriminatory or purely automated employment decisions.
st.markdown("""
<div class="advisory-box">
    <strong>Ethical AI & Decision-Support Notice:</strong><br>
    Predictions generated by this model are probabilistic statistical indicators derived from historical workforce patterns. 
    They <strong>do not establish causality</strong> and must <strong>never</strong> be used as the sole or automated basis 
    for termination, compensation, promotion, or adverse employment actions. All outputs are intended strictly to support 
    human-led talent retention discussions and workload management.
</div>
""", unsafe_allow_html=True)

# Verify model availability before rendering the input form.
# If the artifact is missing, display a clear user-friendly error and halt execution cleanly.
pipeline, model_err = load_trained_pipeline()
if model_err:
    st.error(f"System Error: {model_err}")
    st.info("Please ensure that 'employee_attrition_model_80_20.joblib' is present in the application directory.")
    st.stop()  # Prevents rendering form inputs when the backend model cannot evaluate them

# -----------------------------------------------------------------------------
# 4. Form Input Controls (Organized into Logical Enterprise Sections)
# -----------------------------------------------------------------------------
# Why st.form is used:
# In Streamlit, changing any normal widget (slider, selectbox) immediately reruns
# the entire script. Using `with st.form(...)` batches all 30 user inputs together.
# The script will only process inputs when the user explicitly clicks the submit button.
st.markdown("### Employee Profile Data Collection")
st.markdown("<p style='color: #6B7280; font-size: 0.875rem; margin-top: -0.5rem; margin-bottom: 1.25rem;'>Enter the employee's demographic, departmental, compensation, and satisfaction attributes to evaluate turnover risk.</p>", unsafe_allow_html=True)

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

    # Form Submission
    submit_button = st.form_submit_button(label="Evaluate Attrition Risk Indicator")

# -----------------------------------------------------------------------------
# 5. Validation, Inference & Results Rendering
# -----------------------------------------------------------------------------
# This entire block executes ONLY when the user clicks the "Evaluate" button.
if submit_button:
    # -------------------------------------------------------------------------
    # A. Logical Consistency Validations
    # -------------------------------------------------------------------------
    # Why these checks exist:
    # Although individual inputs might be within numeric range, combinations can
    # be physically impossible in the real world (e.g., an employee cannot have
    # worked at a company for 10 years if their total career experience is only 5 years).
    # We collect warnings and display them to the user without hard-crashing the app.
    validation_warnings = []
    if years_at_company > total_working_years:
        validation_warnings.append("Years at Company cannot exceed Total Working Years.")
    if years_in_current_role > years_at_company:
        validation_warnings.append("Years in Current Role cannot exceed Years at Company.")
    if years_with_curr_manager > years_at_company:
        validation_warnings.append("Years with Current Manager cannot exceed Years at Company.")
    if years_since_last_promotion > years_at_company:
        validation_warnings.append("Years Since Last Promotion cannot exceed Years at Company.")

    # Surface any consistency warnings to help the user correct data entry mistakes
    if validation_warnings:
        for warn in validation_warnings:
            st.warning(f"Input Verification Note: {warn}")

    # -------------------------------------------------------------------------
    # B. Feature Engineering with Zero-Denominator Protection
    # -------------------------------------------------------------------------
    # The trained model requires three engineered tenure ratios that capture
    # career progression and organizational stability.
    #
    # Zero-Division Safety:
    # If an employee has 0 TotalWorkingYears or 0 YearsAtCompany, standard division
    # would crash with ZeroDivisionError (or generate NaN / infinity in NumPy).
    # We defensively assign 0.0, exactly matching the training script's rule.
    
    # Ratio 1: Proportion of total career spent at the current company
    company_exp_ratio = (
        float(years_at_company) / float(total_working_years)
        if total_working_years > 0 else 0.0
    )
    # Ratio 2: Proportion of company tenure spent in the current role
    current_role_ratio = (
        float(years_in_current_role) / float(years_at_company)
        if years_at_company > 0 else 0.0
    )
    # Ratio 3: Proportion of company tenure spent with the current manager
    manager_ratio = (
        float(years_with_curr_manager) / float(years_at_company)
        if years_at_company > 0 else 0.0
    )

    # -------------------------------------------------------------------------
    # C. Assemble 33-Feature Input DataFrame
    # -------------------------------------------------------------------------
    # Scikit-Learn's ColumnTransformer relies on exact feature names to route
    # columns to the correct numeric or categorical transformer.
    # We construct a 1-row Pandas DataFrame containing all 26 numerical (23 direct
    # + 3 engineered) and 7 categorical features with explicit type casting.
    input_dict = {
        # 26 Numerical Features (23 direct + 3 engineered)
        "Age": [int(age)],
        "DailyRate": [int(daily_rate)],
        "DistanceFromHome": [int(distance_from_home)],
        "Education": [int(education)],
        "EnvironmentSatisfaction": [int(environment_satisfaction)],
        "HourlyRate": [int(hourly_rate)],
        "JobInvolvement": [int(job_involvement)],
        "JobLevel": [int(job_level)],
        "JobSatisfaction": [int(job_satisfaction)],
        "MonthlyIncome": [float(monthly_income)],
        "MonthlyRate": [int(monthly_rate)],
        "NumCompaniesWorked": [int(num_companies_worked)],
        "PercentSalaryHike": [int(percent_salary_hike)],
        "PerformanceRating": [int(performance_rating)],
        "RelationshipSatisfaction": [int(relationship_satisfaction)],
        "StockOptionLevel": [int(stock_option_level)],
        "TotalWorkingYears": [int(total_working_years)],
        "TrainingTimesLastYear": [int(training_times_last_year)],
        "WorkLifeBalance": [int(work_life_balance)],
        "YearsAtCompany": [int(years_at_company)],
        "YearsInCurrentRole": [int(years_in_current_role)],
        "YearsSinceLastPromotion": [int(years_since_last_promotion)],
        "YearsWithCurrManager": [int(years_with_curr_manager)],
        "CompanyExperienceRatio": [float(company_exp_ratio)],
        "CurrentRoleTenureRatio": [float(current_role_ratio)],
        "ManagerTenureRatio": [float(manager_ratio)],
        # 7 Categorical Features
        "BusinessTravel": [str(business_travel)],
        "Department": [str(department)],
        "EducationField": [str(education_field)],
        "Gender": [str(gender)],
        "JobRole": [str(job_role)],
        "MaritalStatus": [str(marital_status)],
        "OverTime": [str(over_time)]
    }

    input_df = pd.DataFrame(input_dict)

    # -------------------------------------------------------------------------
    # D. Model Inference & Probability Extraction
    # -------------------------------------------------------------------------
    try:
        # Dynamic positive class mapping:
        # In Scikit-Learn, `predict_proba()` returns a matrix of shape (n_samples, n_classes).
        # We must never assume the positive class ('Attrition = Yes' / 1) is at index 1.
        # By dynamically checking `pipeline.classes_`, we locate the exact column index
        # corresponding to class 1, making our code completely robust to internal ordering.
        model_classes = list(pipeline.classes_)
        if 1 not in model_classes:
            raise ValueError(f"Positive class (1) not found in model classes: {model_classes}")
        
        pos_class_idx = model_classes.index(1)
        
        # Execute prediction and probability estimation
        raw_pred = pipeline.predict(input_df)[0]
        prob_matrix = pipeline.predict_proba(input_df)
        prob_attrition = float(prob_matrix[0, pos_class_idx])
        
        # Determine classification status based on the standard 0.50 (50%) decision threshold
        is_elevated_risk = bool(prob_attrition >= 0.50)
        
        # ---------------------------------------------------------------------
        # E. Render Enterprise Diagnostic Card
        # ---------------------------------------------------------------------
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

        # Statistical Interpretation Note
        st.markdown("""
        <div style="margin-top: 1rem; padding: 0.85rem 1rem; background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 4px; font-size: 0.8125rem; color: #4B5563; line-height: 1.45;">
            <strong>Statistical Interpretation Note:</strong> This percentage represents an algorithmic estimate derived from historical patterns in the IBM HR Analytics dataset. It is a decision-support indicator, not a guaranteed outcome or a definitively calibrated individual probability. It must not be interpreted as certainty that an employee will stay or leave.
        </div>
        """, unsafe_allow_html=True)

        # Contextual Metric Cards
        st.markdown("<h4 style='margin-top: 1.5rem;'>Computed Trajectory Ratios & Context</h4>", unsafe_allow_html=True)
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        with rcol1:
            st.metric("Company Tenure Ratio", f"{company_exp_ratio:.2f}", help="YearsAtCompany / TotalWorkingYears")
        with rcol2:
            st.metric("Role Stability Ratio", f"{current_role_ratio:.2f}", help="YearsInCurrentRole / YearsAtCompany")
        with rcol3:
            st.metric("Manager Alignment Ratio", f"{manager_ratio:.2f}", help="YearsWithCurrManager / YearsAtCompany")
        with rcol4:
            st.metric("Overtime Impact Flag", str(over_time), help="High correlation factor identified in EDA")

        # Responsible HR Review Guidelines (Strictly Non-Prescriptive)
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

    except Exception as e:
        st.error(f"Inference Execution Error: {str(e)}")
        st.info("Please verify that all inputs conform to expected numerical and categorical categories.")

# -----------------------------------------------------------------------------
# 6. Technical Footer & Governance Specification
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8125rem; color: #9CA3AF;">
    <div><strong>Model Pipeline:</strong> Tuned Logistic Regression (Scikit-Learn 1.6.1) | 80:20 Stratified Partition</div>
    <div><strong>Build:</strong> Local Development & Evaluation (Model Version: 80_20 v1)</div>
</div>
""", unsafe_allow_html=True)
