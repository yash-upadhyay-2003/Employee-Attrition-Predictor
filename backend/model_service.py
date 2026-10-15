"""
Model Inference & Feature Engineering Service
=============================================
This module encapsulates the trained Scikit-Learn pipeline for the FastAPI backend.
It handles defensive artifact loading, server-side feature engineering with
zero-denominator protection, tabular DataFrame assembly, and dynamic positive-class
probability extraction.
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from backend.schemas import (
    EmployeePredictionRequest,
    EmployeePredictionResponse,
    ComputedRatios,
    ModelInfoResponse,
    HealthResponse
)

# Robust path resolution: resolve root directory relative to this file's location
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "employee_attrition_model_80_20.joblib"
RESULTS_JSON_PATH = PROJECT_ROOT / "retraining_results.json"


class ModelService:
    """
    Singleton service managing model artifact loading and inference execution.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        self.pipeline: Optional[Any] = None
        self._load_error: Optional[str] = None
        self._results_metadata: Dict[str, Any] = {}
        
        # Load metadata and model
        self._load_results_metadata()
        self.load_model()

    def _load_results_metadata(self) -> None:
        """Safely load verified training results from retraining_results.json if present."""
        if RESULTS_JSON_PATH.exists():
            try:
                with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                    self._results_metadata = json.load(f)
            except Exception:
                self._results_metadata = {}

    def load_model(self) -> None:
        """
        Load the trained Scikit-Learn pipeline from disk with defensive validation.
        Fails safely if the artifact is missing; NEVER trains a replacement model.
        """
        if not self.model_path.exists():
            self._load_error = f"Model artifact '{self.model_path.name}' not found at {self.model_path}."
            self.pipeline = None
            return

        try:
            loaded_obj = joblib.load(self.model_path)
            # Verify required estimator interface
            if not hasattr(loaded_obj, "predict") or not hasattr(loaded_obj, "predict_proba"):
                raise TypeError(f"Loaded object from {self.model_path} does not implement predict and predict_proba.")
            if not hasattr(loaded_obj, "classes_"):
                raise AttributeError("Loaded estimator does not define classes_ attribute.")
            
            self.pipeline = loaded_obj
            self._load_error = None
        except Exception as e:
            self._load_error = f"Failed to load model artifact: {str(e)}"
            self.pipeline = None

    @property
    def is_ready(self) -> bool:
        """Returns True if the pipeline is successfully loaded in memory."""
        return self.pipeline is not None

    def get_health(self) -> HealthResponse:
        """Returns minimal operational health details without exposing sensitive filesystem paths."""
        return HealthResponse(
            status="healthy" if self.is_ready else "degraded",
            model_loaded=self.is_ready,
            version="1.0.0",
            environment={
                "python_version": sklearn.__version__,
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__
            }
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Returns verified pipeline metadata and metrics."""
        if not self.is_ready:
            raise RuntimeError(f"Model service is not initialized: {self._load_error}")

        preprocessor = self.pipeline.named_steps.get("preprocessor")
        num_features = []
        cat_features = []
        if preprocessor and hasattr(preprocessor, "transformers_"):
            num_features = preprocessor.transformers_[0][2]
            cat_features = preprocessor.transformers_[1][2]

        cv_f1 = self._results_metadata.get("selected_cv_f1")
        cv_auc = self._results_metadata.get("selected_cv_roc_auc")

        return ModelInfoResponse(
            model_name=str(self._results_metadata.get("selected_model", "Logistic Regression (Tuned)")),
            artifact_filename=self.model_path.name,
            partition_strategy="80:20 Stratified Split",
            total_features=len(num_features) + len(cat_features) if num_features else 33,
            numerical_features_count=len(num_features) if num_features else 26,
            categorical_features_count=len(cat_features) if cat_features else 7,
            classes=[int(c) for c in self.pipeline.classes_],
            decision_threshold=0.50,
            validation_cv_f1=round(float(cv_f1), 4) if cv_f1 is not None else None,
            validation_cv_roc_auc=round(float(cv_auc), 4) if cv_auc is not None else None
        )

    def predict(self, req: EmployeePredictionRequest) -> EmployeePredictionResponse:
        """
        Execute feature engineering, DataFrame assembly, and model inference.

        Parameters:
        -----------
        req: Validated 30-field EmployeePredictionRequest.

        Returns:
        --------
        EmployeePredictionResponse with prediction, probabilities, ratios, and neutral status text.
        """
        if not self.is_ready:
            raise RuntimeError(f"Model inference unavailable: {self._load_error}")

        # ---------------------------------------------------------------------
        # 1. Non-blocking consistency validations
        # ---------------------------------------------------------------------
        warnings: List[str] = []
        if req.years_at_company > req.total_working_years:
            warnings.append("Years at Company cannot exceed Total Working Years.")
        if req.years_in_current_role > req.years_at_company:
            warnings.append("Years in Current Role cannot exceed Years at Company.")
        if req.years_with_curr_manager > req.years_at_company:
            warnings.append("Years with Current Manager cannot exceed Years at Company.")
        if req.years_since_last_promotion > req.years_at_company:
            warnings.append("Years Since Last Promotion cannot exceed Years at Company.")

        # ---------------------------------------------------------------------
        # 2. Server-side Feature Engineering with Zero-Denominator Resilience
        # -------------------------------------------------------------------------
        # Formula 1: Proportion of total career spent at the current company
        company_exp_ratio = (
            float(req.years_at_company) / float(req.total_working_years)
            if req.total_working_years > 0 else 0.0
        )
        # Formula 2: Proportion of company tenure spent in current role
        current_role_ratio = (
            float(req.years_in_current_role) / float(req.years_at_company)
            if req.years_at_company > 0 else 0.0
        )
        # Formula 3: Proportion of company tenure spent with current manager
        manager_ratio = (
            float(req.years_with_curr_manager) / float(req.years_at_company)
            if req.years_at_company > 0 else 0.0
        )

        computed_ratios = ComputedRatios(
            company_experience_ratio=round(company_exp_ratio, 4),
            current_role_tenure_ratio=round(current_role_ratio, 4),
            manager_tenure_ratio=round(manager_ratio, 4)
        )

        # ---------------------------------------------------------------------
        # 3. Assemble 33-Feature DataFrame with Exact Preprocessor Schema
        # ---------------------------------------------------------------------
        input_dict = {
            # 26 Numerical Features (23 raw + 3 engineered)
            "Age": [int(req.age)],
            "DailyRate": [int(req.daily_rate)],
            "DistanceFromHome": [int(req.distance_from_home)],
            "Education": [int(req.education)],
            "EnvironmentSatisfaction": [int(req.environment_satisfaction)],
            "HourlyRate": [int(req.hourly_rate)],
            "JobInvolvement": [int(req.job_involvement)],
            "JobLevel": [int(req.job_level)],
            "JobSatisfaction": [int(req.job_satisfaction)],
            "MonthlyIncome": [float(req.monthly_income)],
            "MonthlyRate": [int(req.monthly_rate)],
            "NumCompaniesWorked": [int(req.num_companies_worked)],
            "PercentSalaryHike": [int(req.percent_salary_hike)],
            "PerformanceRating": [int(req.performance_rating)],
            "RelationshipSatisfaction": [int(req.relationship_satisfaction)],
            "StockOptionLevel": [int(req.stock_option_level)],
            "TotalWorkingYears": [int(req.total_working_years)],
            "TrainingTimesLastYear": [int(req.training_times_last_year)],
            "WorkLifeBalance": [int(req.work_life_balance)],
            "YearsAtCompany": [int(req.years_at_company)],
            "YearsInCurrentRole": [int(req.years_in_current_role)],
            "YearsSinceLastPromotion": [int(req.years_since_last_promotion)],
            "YearsWithCurrManager": [int(req.years_with_curr_manager)],
            "CompanyExperienceRatio": [float(company_exp_ratio)],
            "CurrentRoleTenureRatio": [float(current_role_ratio)],
            "ManagerTenureRatio": [float(manager_ratio)],
            # 7 Categorical Features
            "BusinessTravel": [str(req.business_travel)],
            "Department": [str(req.department)],
            "EducationField": [str(req.education_field)],
            "Gender": [str(req.gender)],
            "JobRole": [str(req.job_role)],
            "MaritalStatus": [str(req.marital_status)],
            "OverTime": [str(req.over_time)]
        }

        input_df = pd.DataFrame(input_dict)

        # ---------------------------------------------------------------------
        # 4. Pipeline Inference & Dynamic Probability Extraction
        # ---------------------------------------------------------------------
        model_classes = list(self.pipeline.classes_)
        if 1 not in model_classes:
            raise ValueError(f"Positive class (1) not found in model classes: {model_classes}")
        
        pos_class_idx = model_classes.index(1)
        neg_class_idx = model_classes.index(0)

        # Execute predictions natively through pipeline
        raw_pred = int(self.pipeline.predict(input_df)[0])
        prob_matrix = self.pipeline.predict_proba(input_df)

        prob_attrition = float(prob_matrix[0, pos_class_idx])
        prob_retention = float(prob_matrix[0, neg_class_idx])

        # Decision threshold = 0.50
        decision_threshold = 0.50
        is_elevated_risk = bool(prob_attrition >= decision_threshold)

        # Neutral, non-prescriptive wording matching app.py
        status_text = (
            "Elevated Model-Estimated Attrition Probability"
            if is_elevated_risk
            else "Lower Model-Estimated Attrition Probability"
        )
        predicted_label: Literal["No", "Yes"] = "Yes" if raw_pred == 1 else "No"

        return EmployeePredictionResponse(
            prediction=raw_pred,
            predicted_label=predicted_label,
            probability_attrition=round(prob_attrition, 4),
            probability_retention=round(prob_retention, 4),
            is_elevated_risk=is_elevated_risk,
            status_text=status_text,
            decision_threshold=decision_threshold,
            computed_ratios=computed_ratios,
            warnings=warnings
        )


# Global singleton instance for the FastAPI application
model_service = ModelService()
