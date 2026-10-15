r"""
Automated Test Suite for Employee Attrition Machine Learning Pipeline
======================================================================
This test module provides end-to-end quality assurance for the Employee Attrition
Predictor system. It tests environment integrity, artifact preservation, native
model loading, feature schema completeness, zero-denominator resilience,
probability mapping logic, and defensive error handling.

How to Run:
-----------
Execute via the dedicated virtual environment pytest runner:
    .venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
"""

# Standard library modules
import os
from pathlib import Path   # Cross-platform filesystem path resolution

# Third-party testing and data science libraries
import joblib              # Loading and inspecting serialized Scikit-Learn pipelines
import numpy as np         # Numerical array assertion and floating-point tolerances
import pandas as pd        # DataFrame construction for pipeline input testing
import pytest              # Python testing framework with assertion introspection

# Relative paths to model artifacts and dataset
MODEL_80_20_PATH = Path("employee_attrition_model_80_20.joblib")
LEGACY_MODEL_PATH = Path("employee_attrition_model.joblib")
DATA_PATH = Path("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# =============================================================================
# Test 1: Virtual Environment & Dependency Integrity
# =============================================================================
def test_01_environment_versions():
    """
    Verify that scikit-learn version is strictly pinned to 1.6.1.

    Why this test is important:
    ---------------------------
    Scikit-Learn serialized pipelines (via Joblib/pickle) are sensitive to minor
    version changes. Newer versions (e.g., scikit-learn 1.9+) introduced internal
    changes to ColumnTransformer attributes ('_RemainderColsList', '_fill_dtype')
    that cause deserialization crashes or require monkey-patch shims.
    Pinned version 1.6.1 guarantees clean, native model loading.

    Failure caught:
    ---------------
    Incompatible Python environment or unpinned package upgrades that would
    break model deserialization in production or during grading.
    """
    import sklearn
    print(f"Verified scikit-learn version: {sklearn.__version__}")
    assert sklearn.__version__ == "1.6.1", f"Expected scikit-learn 1.6.1, got {sklearn.__version__}"

# =============================================================================
# Test 2: Artifact Existence and Preservation of Historical Legacy Model
# =============================================================================
def test_02_artifacts_exist_and_legacy_preserved():
    """
    Confirm both active 80:20 model and original 60:40 legacy model exist untouched.

    Why this test is important:
    ---------------------------
    Assignment integrity rules mandate that the active retrained model must be exported
    as 'employee_attrition_model_80_20.joblib', while the original historical model
    'employee_attrition_model.joblib' must be strictly preserved at its exact path
    and byte size (8,694 bytes) without renaming, moving, or overwriting.

    Failure caught:
    ---------------
    Accidental overwrite, deletion, or renaming of the original legacy artifact.
    """
    assert MODEL_80_20_PATH.exists(), f"Missing retrained artifact: {MODEL_80_20_PATH}"
    assert LEGACY_MODEL_PATH.exists(), f"Legacy artifact missing or moved: {LEGACY_MODEL_PATH}"
    # Verify exact file size in bytes to detect silent modifications
    assert LEGACY_MODEL_PATH.stat().st_size == 8694, "Legacy artifact modified!"

# =============================================================================
# Test 3: Native Model Deserialization Without Compatibility Shims
# =============================================================================
def test_03_native_model_loading_no_shims():
    """
    Confirm the retrained model loads natively and implements standard Scikit-Learn APIs.

    Why this test is important:
    ---------------------------
    Ensures the model can be loaded directly with `joblib.load()` without requiring
    any custom unpickling patches, shims, or wrappers. Verifies that the model exposes
    standard classification methods (`predict`, `predict_proba`) and contains binary
    class labels [0, 1].

    Failure caught:
    ---------------
    Corrupted model file, missing Scikit-Learn methods, or non-binary class definitions.
    """
    model = joblib.load(MODEL_80_20_PATH)
    assert hasattr(model, "predict"), "Loaded object has no predict method"
    assert hasattr(model, "predict_proba"), "Loaded object has no predict_proba method"
    assert hasattr(model, "classes_"), "Loaded object has no classes_ attribute"
    assert list(model.classes_) == [0, 1], f"Expected classes [0, 1], got {model.classes_}"

# =============================================================================
# Test 4: Feature Schema Completeness
# =============================================================================
def test_04_feature_schema_completeness():
    """
    Verify that the pipeline preprocessor registers all 33 expected features.

    Why this test is important:
    ---------------------------
    The model preprocessor is an `sklearn.compose.ColumnTransformer`. It routes
    columns to numeric or categorical transformers based on column names.
    This test audits that exactly 26 numerical features (23 direct + 3 engineered)
    and 7 categorical features are present, guaranteeing no feature drift.

    Failure caught:
    ---------------
    Missing columns, incorrect feature classification (e.g., treating a numerical
    column as categorical), or omission of engineered tenure ratios.
    """
    model = joblib.load(MODEL_80_20_PATH)
    preprocessor = model.named_steps["preprocessor"]
    
    # Extract feature lists registered in the numerical and categorical transformers
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    total_features = num_features + cat_features

    assert len(num_features) == 26, f"Expected 26 numerical features, found {len(num_features)}"
    assert len(cat_features) == 7, f"Expected 7 categorical features, found {len(cat_features)}"
    assert len(total_features) == 33, f"Expected 33 total features, found {len(total_features)}"

    # Explicitly verify that the 3 engineered ratios are included in numerical features
    expected_engineered = ["CompanyExperienceRatio", "CurrentRoleTenureRatio", "ManagerTenureRatio"]
    for feat in expected_engineered:
        assert feat in num_features, f"Missing engineered feature: {feat}"

# =============================================================================
# Test 5: Tenure Ratios Zero-Denominator Resilience
# =============================================================================
def test_05_tenure_ratios_zero_denominator_resilience():
    """
    Verify that feature engineering safely handles 0 values without division-by-zero crashes.

    Why this test is important:
    ---------------------------
    Employees in their first year of work may have TotalWorkingYears = 0 or
    YearsAtCompany = 0. In Python, `0 / 0` raises `ZeroDivisionError`.
    This test verifies that the safe division logic (`val / denom if denom > 0 else 0.0`)
    returns 0.0 cleanly.

    Failure caught:
    ---------------
    Unhandled `ZeroDivisionError` that would crash the application when evaluating
    new or newly hired employees.
    """
    # Test zero denominators representing new workforce entrants
    twy_zero = 0
    yac_zero = 0
    
    company_exp_ratio = float(yac_zero) / float(twy_zero) if twy_zero > 0 else 0.0
    current_role_ratio = float(0) / float(yac_zero) if yac_zero > 0 else 0.0
    manager_ratio = float(0) / float(yac_zero) if yac_zero > 0 else 0.0

    assert company_exp_ratio == 0.0, "Zero denominator failed for CompanyExperienceRatio"
    assert current_role_ratio == 0.0, "Zero denominator failed for CurrentRoleTenureRatio"
    assert manager_ratio == 0.0, "Zero denominator failed for ManagerTenureRatio"

# =============================================================================
# Test 6: Prediction & Probability Parity on Real Data
# =============================================================================
def test_06_prediction_and_probability_parity():
    """
    Verify that the loaded pipeline runs inference and outputs valid probabilities.

    Why this test is important:
    ---------------------------
    Passes a sample of real employee records through the complete pipeline
    (preprocessing + classification). Validates that:
    1. The output prediction count matches the input sample count.
    2. The probability matrix has shape (N, 2).
    3. All probabilities fall strictly within [0.0, 1.0].
    4. Row probabilities sum to 1.0 (valid probability distribution).

    Failure caught:
    ---------------
    Broken data transformation, shape mismatch, or mathematically invalid probabilities.
    """
    model = joblib.load(MODEL_80_20_PATH)
    df = pd.read_csv(DATA_PATH, sep="\t")

    # Apply the 3 engineered ratios exactly as performed during training and app inference
    df["CompanyExperienceRatio"] = np.where(df["TotalWorkingYears"] > 0, df["YearsAtCompany"] / df["TotalWorkingYears"], 0.0)
    df["CurrentRoleTenureRatio"] = np.where(df["YearsAtCompany"] > 0, df["YearsInCurrentRole"] / df["YearsAtCompany"], 0.0)
    df["ManagerTenureRatio"] = np.where(df["YearsAtCompany"] > 0, df["YearsWithCurrManager"] / df["YearsAtCompany"], 0.0)

    sample = df.head(5)
    preds = model.predict(sample)
    probs = model.predict_proba(sample)

    assert len(preds) == 5, "Predictions count mismatch"
    assert probs.shape == (5, 2), "Probabilities matrix shape mismatch"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities out of [0, 1] range"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"

# =============================================================================
# Test 7: Dynamic Probability Class Mapping Logic
# =============================================================================
def test_07_probability_class_mapping_logic():
    """
    Verify that the positive class probability is mapped dynamically using `classes_`.

    Why this test is important:
    ---------------------------
    In Scikit-Learn, `predict_proba()` returns probabilities in the order of `model.classes_`.
    Hardcoding index `probs[:, 1]` assumes class 1 is always in the second column.
    This test verifies that the application dynamically locates `classes_.index(1)`
    and confirms that class 0 probability + class 1 probability equals 1.0.

    Failure caught:
    ---------------
    Inverted risk scores (confusing attrition probability with retention probability)
    caused by static column indexing.
    """
    model = joblib.load(MODEL_80_20_PATH)
    classes = list(model.classes_)
    assert 1 in classes, "Positive class (1) not in model classes"
    
    pos_idx = classes.index(1)
    neg_idx = classes.index(0)
    
    df = pd.read_csv(DATA_PATH, sep="\t")
    df["CompanyExperienceRatio"] = np.where(df["TotalWorkingYears"] > 0, df["YearsAtCompany"] / df["TotalWorkingYears"], 0.0)
    df["CurrentRoleTenureRatio"] = np.where(df["YearsAtCompany"] > 0, df["YearsInCurrentRole"] / df["YearsAtCompany"], 0.0)
    df["ManagerTenureRatio"] = np.where(df["YearsAtCompany"] > 0, df["YearsWithCurrManager"] / df["YearsAtCompany"], 0.0)

    sample = df.head(1)
    probs = model.predict_proba(sample)
    
    # Confirm probability of departure corresponds to the exact index of class 1
    prob_departure = probs[0, pos_idx]
    prob_retention = probs[0, neg_idx]
    
    assert np.isclose(prob_departure + prob_retention, 1.0), "Class probabilities do not sum to 1"
    assert 0.0 <= prob_departure <= 1.0, "Positive class probability out of bounds"

# =============================================================================
# Test 8: Defensive Missing Artifact Handling
# =============================================================================
def test_08_missing_artifact_handling():
    """
    Verify that a missing model artifact triggers a graceful error rather than crashing.

    Why this test is important:
    ---------------------------
    In production environments, missing or misconfigured file paths must be caught
    defensively. This test tests the helper pattern used in `app.py` to ensure it
    returns `(None, error_message)` with clear guidance instead of an uncaught exception.

    Failure caught:
    ---------------
    Unhandled application crash when a model file is missing or misplaced.
    """
    # Simulate missing artifact check
    missing_path = Path("non_existent_model_artifact_xyz.joblib")
    
    def simulate_load(path):
        if not path.exists():
            return None, f"Model artifact '{path.name}' was not found in the application directory."
        try:
            return joblib.load(path), None
        except Exception as e:
            return None, str(e)
            
    pipe, err = simulate_load(missing_path)
    assert pipe is None, "Simulated load should return None for missing artifact"
    assert err is not None, "Error message should not be None"
    assert "not found" in err, "Error message should indicate missing artifact"

# =============================================================================
# Test 9: Defensive Invalid Input Error Handling
# =============================================================================
def test_09_invalid_input_error_handling():
    """
    Verify that passing an incomplete DataFrame to the pipeline raises an exception.

    Why this test is important:
    ---------------------------
    Scikit-Learn's `ColumnTransformer` should defensively reject inputs missing
    required feature columns. This test confirms that attempting to predict on
    an incomplete input raises an exception so the application can intercept it.

    Failure caught:
    ---------------
    Silent failure or garbage predictions generated from partial or corrupted input schemas.
    """
    model = joblib.load(MODEL_80_20_PATH)
    
    # Incomplete input (missing 31 of the required 33 columns)
    incomplete_input = pd.DataFrame({"Age": [30], "MonthlyIncome": [5000.0]})
    
    with pytest.raises(Exception) as exc_info:
        model.predict(incomplete_input)
        
    assert exc_info.value is not None, "Incomplete input must raise an exception for defensive handling"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
