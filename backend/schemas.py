"""
Pydantic Request & Response Schemas for Employee Attrition API
=============================================================
This module defines strict data models for input validation, schema documentation,
and serialized responses for the FastAPI prediction backend.

Validation Rules:
-----------------
- All 30 raw employee features are strictly typed and range-checked to mirror
  the Streamlit user interface controls exactly.
- Categorical features use Literal types to reject unapproved categories.
- Engineered ratios are NOT accepted from clients; they are calculated server-side.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Allowed categorical values matching the IBM HR dataset and app.py
GenderType = Literal["Male", "Female"]
MaritalStatusType = Literal["Married", "Single", "Divorced"]
DepartmentType = Literal["Research & Development", "Sales", "Human Resources"]
EducationFieldType = Literal[
    "Life Sciences", "Medical", "Marketing",
    "Technical Degree", "Human Resources", "Other"
]
BusinessTravelType = Literal["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
OverTimeType = Literal["No", "Yes"]
JobRoleType = Literal[
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]


class EmployeePredictionRequest(BaseModel):
    """
    Request model containing the 30 raw employee attributes collected from users.
    
    Field aliases allow both camelCase/PascalCase (matching dataset) and snake_case
    (standard Python JSON format).
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Demographics & Alignment
    age: int = Field(..., ge=18, le=65, alias="Age", description="Employee chronological age in years [18, 65].")
    gender: GenderType = Field(..., alias="Gender", description="Employee gender ('Male' or 'Female').")
    marital_status: MaritalStatusType = Field(..., alias="MaritalStatus", description="Civil marital status.")
    department: DepartmentType = Field(..., alias="Department", description="Organizational department.")
    job_role: JobRoleType = Field(..., alias="JobRole", description="Specific job role designation.")
    job_level: int = Field(..., ge=1, le=5, alias="JobLevel", description="Job tier from 1 (Entry) to 5 (Executive).")
    education: int = Field(..., ge=1, le=5, alias="Education", description="Education level from 1 (Below College) to 5 (Doctor).")
    education_field: EducationFieldType = Field(..., alias="EducationField", description="Field of academic study.")
    business_travel: BusinessTravelType = Field(..., alias="BusinessTravel", description="Business travel frequency.")

    # Compensation & Pay Structure
    monthly_income: float = Field(..., ge=1000.0, le=20000.0, alias="MonthlyIncome", description="Monthly gross income in USD.")
    stock_option_level: int = Field(..., ge=0, le=3, alias="StockOptionLevel", description="Stock grant level tier [0, 3].")
    daily_rate: int = Field(..., ge=100, le=1500, alias="DailyRate", description="Daily pay rate in USD [100, 1500].")
    hourly_rate: int = Field(..., ge=30, le=100, alias="HourlyRate", description="Hourly pay rate in USD [30, 100].")
    monthly_rate: int = Field(..., ge=2000, le=27000, alias="MonthlyRate", description="Monthly compensation rate in USD [2000, 27000].")
    percent_salary_hike: int = Field(..., ge=10, le=25, alias="PercentSalaryHike", description="Percent salary hike on last review [10, 25].")

    # Career History & Tenure
    total_working_years: int = Field(..., ge=0, le=40, alias="TotalWorkingYears", description="Total years of work experience [0, 40].")
    years_at_company: int = Field(..., ge=0, le=40, alias="YearsAtCompany", description="Tenure at current company in years [0, 40].")
    years_in_current_role: int = Field(..., ge=0, le=25, alias="YearsInCurrentRole", description="Years in current role [0, 25].")
    years_with_curr_manager: int = Field(..., ge=0, le=25, alias="YearsWithCurrManager", description="Years under current manager [0, 25].")
    years_since_last_promotion: int = Field(..., ge=0, le=15, alias="YearsSinceLastPromotion", description="Years since last promotion [0, 15].")
    num_companies_worked: int = Field(..., ge=0, le=10, alias="NumCompaniesWorked", description="Number of previous employers [0, 10].")

    # Workplace Environment & Satisfaction
    over_time: OverTimeType = Field(..., alias="OverTime", description="Contracted overtime ('No' or 'Yes').")
    distance_from_home: int = Field(..., ge=1, le=30, alias="DistanceFromHome", description="Commute distance in kilometers [1, 30].")
    training_times_last_year: int = Field(..., ge=0, le=6, alias="TrainingTimesLastYear", description="Training sessions attended last year [0, 6].")
    job_satisfaction: int = Field(..., ge=1, le=4, alias="JobSatisfaction", description="Job satisfaction rating [1: Low, 4: Very High].")
    environment_satisfaction: int = Field(..., ge=1, le=4, alias="EnvironmentSatisfaction", description="Work environment satisfaction [1: Low, 4: Very High].")
    relationship_satisfaction: int = Field(..., ge=1, le=4, alias="RelationshipSatisfaction", description="Workplace relationship satisfaction [1: Low, 4: Very High].")
    work_life_balance: int = Field(..., ge=1, le=4, alias="WorkLifeBalance", description="Work-life balance rating [1: Bad, 4: Best].")
    job_involvement: int = Field(..., ge=1, le=4, alias="JobInvolvement", description="Job involvement rating [1: Low, 4: Very High].")
    performance_rating: Literal[3, 4] = Field(..., alias="PerformanceRating", description="Performance evaluation rating (3: Excellent, 4: Outstanding).")


class ComputedRatios(BaseModel):
    """Derived tenure ratios computed server-side with zero-denominator resilience."""
    company_experience_ratio: float = Field(..., description="YearsAtCompany / TotalWorkingYears (0.0 if TotalWorkingYears <= 0).")
    current_role_tenure_ratio: float = Field(..., description="YearsInCurrentRole / YearsAtCompany (0.0 if YearsAtCompany <= 0).")
    manager_tenure_ratio: float = Field(..., description="YearsWithCurrManager / YearsAtCompany (0.0 if YearsAtCompany <= 0).")


class EmployeePredictionResponse(BaseModel):
    """Complete prediction response contract."""
    prediction: int = Field(..., description="Predicted class index (0 for retained, 1 for departed).")
    predicted_label: Literal["No", "Yes"] = Field(..., description="Human-readable prediction label.")
    probability_attrition: float = Field(..., description="Model-estimated probability of departure (positive class 1).")
    probability_retention: float = Field(..., description="Model-estimated probability of retention (negative class 0).")
    is_elevated_risk: bool = Field(..., description="True if probability_attrition >= decision_threshold (0.50).")
    status_text: str = Field(..., description="Neutral diagnostic status string.")
    decision_threshold: float = Field(default=0.50, description="Classification decision threshold boundary.")
    computed_ratios: ComputedRatios = Field(..., description="Server-computed tenure ratios.")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking domain consistency warnings.")


class HealthResponse(BaseModel):
    """Health probe response model."""
    status: Literal["healthy", "degraded"] = Field(..., description="API operational health status.")
    model_loaded: bool = Field(..., description="True if model artifact is successfully loaded in memory.")
    version: str = Field(default="1.0.0", description="API version identifier.")
    environment: Dict[str, str] = Field(default_factory=dict, description="Safe runtime version details.")


class ModelInfoResponse(BaseModel):
    """Model metadata response model."""
    model_name: str = Field(..., description="Winning model architecture name.")
    artifact_filename: str = Field(..., description="Serialized pipeline artifact filename.")
    partition_strategy: str = Field(..., description="Train/test partition methodology.")
    total_features: int = Field(..., description="Total input features required by ColumnTransformer.")
    numerical_features_count: int = Field(..., description="Number of numerical features transformed.")
    categorical_features_count: int = Field(..., description="Number of categorical features one-hot encoded.")
    classes: List[int] = Field(..., description="Model class labels [0, 1].")
    decision_threshold: float = Field(default=0.50, description="Decision boundary threshold.")
    validation_cv_f1: Optional[float] = Field(None, description="5-fold CV F1 score on training partition.")
    validation_cv_roc_auc: Optional[float] = Field(None, description="5-fold CV ROC-AUC score on training partition.")
