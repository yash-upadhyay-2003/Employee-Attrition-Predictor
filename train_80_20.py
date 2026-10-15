"""
Model Retraining & Evaluation Pipeline (Stratified 80:20 Partition)
===================================================================
This script executes the complete machine learning training and evaluation lifecycle
for the Employee Attrition Predictor project, strictly adhering to assignment protocols.

End-to-End Training Flow:
-------------------------
1. Data Ingestion: Loads the tab-separated IBM HR Analytics dataset (1,470 rows, 35 columns).
2. Feature Engineering: Generates 3 tenure trajectory ratios with zero-division protection.
3. Feature Selection: Removes target ('Attrition'), employee identifier ('EmployeeNumber'),
   and 3 constant columns ('EmployeeCount', 'Over18', 'StandardHours') to prevent leakage and noise.
4. Stratified 80:20 Split: Divides the dataset into training (1,176 rows) and test (294 rows)
   partitions, preserving the ~16.1% positive class proportion in both subsets.
5. Leakage-Free Preprocessing: Configures a ColumnTransformer with SimpleImputer,
   StandardScaler, and OneHotEncoder fitted EXCLUSIVELY on X_train.
6. Model Comparison & Tuning: Trains baseline and tuned variants of Decision Tree,
   Random Forest, and Logistic Regression using 5-Fold Stratified Cross-Validation on X_train.
7. Model Selection: Selects the winning model based on positive-class Attrition F1-score
   (with ROC-AUC as a tie-breaker for differences < 0.01).
8. Held-Out Evaluation: Evaluates all models once on the held-out test partition.
9. Artifact Serialization & Verification: Exports the winning pipeline to disk using Joblib
   and verifies prediction and probability parity upon reloading.
"""

# Standard library modules
import json                # Export evaluation metrics and summary statistics to a structured JSON file
from pathlib import Path   # Robust, cross-platform path resolution

# Core scientific computing and data analysis libraries
import joblib              # Serializing and saving trained Scikit-Learn pipelines
import numpy as np         # Numerical operations, array handling, and conditional assignment (np.where)
import pandas as pd        # Tabular data manipulation, DataFrame slicing, and categorical frequency mapping
import matplotlib.pyplot as plt  # Generating evaluation figures (ROC curves and feature importance charts)

# Scikit-Learn: Cross-validation, hyperparameter tuning, and dataset partitioning
from sklearn.model_selection import (
    train_test_split,     # Splits dataset into training and testing subsets
    StratifiedKFold,      # Generates cross-validation folds preserving class ratios
    cross_validate,       # Evaluates multiple metrics across cross-validation folds
    GridSearchCV          # Exhaustive search over specified hyperparameter values for an estimator
)

# Scikit-Learn: Preprocessing and pipeline encapsulation
from sklearn.compose import ColumnTransformer  # Applies distinct transformations to numerical vs categorical columns
from sklearn.pipeline import Pipeline          # Chains preprocessing steps sequentially with a model estimator
from sklearn.impute import SimpleImputer       # Replaces missing values with median or most-frequent strategy
from sklearn.preprocessing import (
    StandardScaler,       # Standardizes numerical features by removing the mean and scaling to unit variance
    OneHotEncoder         # Encodes categorical features as one-hot numeric arrays
)

# Scikit-Learn: Supervised classification algorithms
from sklearn.tree import DecisionTreeClassifier       # Non-parametric decision tree model
from sklearn.ensemble import RandomForestClassifier   # Ensemble bagging model of randomized decision trees
from sklearn.linear_model import LogisticRegression   # Linear classification model with probability estimation

# Scikit-Learn: Evaluation metrics for binary classification
from sklearn.metrics import (
    accuracy_score,        # Proportion of correctly classified samples
    precision_score,       # True Positives / (True Positives + False Positives): avoids false alarms
    recall_score,          # True Positives / (True Positives + False Negatives): catches departures
    f1_score,              # Harmonic mean of precision and recall for the positive minority class
    roc_auc_score,         # Area Under the Receiver Operating Characteristic curve
    confusion_matrix,      # Matrix comparing actual vs predicted class labels
    classification_report, # Formatted text summary of precision, recall, and F1 per class
    roc_curve              # Computes False Positive Rate and True Positive Rate coordinates for plotting
)

def run_retraining():
    """
    Execute the deterministic 80:20 retraining, hyperparameter tuning, model selection,
    held-out testing, plot generation, and artifact export workflow.

    Why this function exists:
    -------------------------
    This function consolidates the entire machine learning pipeline into a single,
    reproducible script. By running all steps deterministically with fixed random seeds
    (random_state=42), it guarantees that data splits, cross-validation folds, and
    model weights are completely reproducible across environments.
    """
    print("=" * 70)
    print("STAGE 3: RETRAINING ON STRATIFIED 80:20 SPLIT")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    # Why sep="\t": The IBM HR Analytics CSV file in this repository uses tab delimiters
    # rather than standard commas. Using standard read_csv() without sep="\t" would load
    # the entire row into a single column.
    data_path = Path("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = pd.read_csv(data_path, sep="\t")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # -------------------------------------------------------------------------
    # 2. Feature Engineering & Column Pruning
    # -------------------------------------------------------------------------
    # We engineer 3 tenure ratios that capture relative career stability.
    #
    # Zero-Denominator Handling:
    # Employees who have just entered the workforce may have TotalWorkingYears = 0
    # or YearsAtCompany = 0. In Python, division by zero raises ZeroDivisionError,
    # and in NumPy it produces NaN/inf. We use np.where() to assign 0.0 when
    # the denominator is zero.
    
    # Ratio 1: Fraction of total career spent at the current company
    df["CompanyExperienceRatio"] = np.where(
        df["TotalWorkingYears"] > 0,
        df["YearsAtCompany"] / df["TotalWorkingYears"],
        0.0
    )
    # Ratio 2: Fraction of company tenure spent in current role
    df["CurrentRoleTenureRatio"] = np.where(
        df["YearsAtCompany"] > 0,
        df["YearsInCurrentRole"] / df["YearsAtCompany"],
        0.0
    )
    # Ratio 3: Fraction of company tenure spent under current manager
    df["ManagerTenureRatio"] = np.where(
        df["YearsAtCompany"] > 0,
        df["YearsWithCurrManager"] / df["YearsAtCompany"],
        0.0
    )

    # Columns to drop before training:
    # - Attrition: This is our target label (y). Must be separated from features (X) to prevent leakage.
    # - EmployeeNumber: Arbitrary employee ID number; contains no generalizable predictive pattern.
    # - EmployeeCount: Constant value (1 across all rows). Zero variance provides zero information.
    # - Over18: Constant value ('Y' across all rows). Zero variance provides zero information.
    # - StandardHours: Constant value (80 across all rows). Zero variance provides zero information.
    # - AgeGroup: If previously created during exploratory analysis, drop it to avoid redundancy with 'Age'.
    columns_to_drop = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]
    if "AgeGroup" in df.columns:
        columns_to_drop.append("AgeGroup")

    X = df.drop(columns=columns_to_drop)
    # Map binary string target labels to integer values: 'No' -> 0 (retained), 'Yes' -> 1 (departed)
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # -------------------------------------------------------------------------
    # 3. Stratified 80:20 Train/Test Partitioning
    # -------------------------------------------------------------------------
    # Why Stratification is Critical:
    # The IBM HR dataset is imbalanced: only ~16.1% of employees departed ('Yes').
    # A standard random split might accidentally assign 20% leavers to the train set
    # and 10% to the test set, creating distribution mismatch.
    # Setting `stratify=y` ensures that both X_train and X_test receive exactly
    # the same ~16.1% proportion of the positive class.
    # `random_state=42` ensures identical partition splits across runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n--- SPLIT VERIFICATION ---")
    print(f"Total rows: {len(X)}")
    print(f"X_train rows: {X_train.shape[0]} (Expected: 1176)")
    print(f"X_test rows:  {X_test.shape[0]} (Expected: 294)")
    assert X_train.shape[0] == 1176, "X_train row count mismatch!"
    assert X_test.shape[0] == 294, "X_test row count mismatch!"

    print("\nTarget Class Distribution in Train:")
    train_counts = y_train.value_counts()
    print(train_counts)
    print(f"Train Attrition Rate: {y_train.mean():.4%}")

    print("\nTarget Class Distribution in Test:")
    test_counts = y_test.value_counts()
    print(test_counts)
    print(f"Test Attrition Rate: {y_test.mean():.4%}")

    # -------------------------------------------------------------------------
    # 4. Leakage-Free Preprocessing Pipeline Architecture
    # -------------------------------------------------------------------------
    # Identifying column data types dynamically from X_train
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"\nNumerical features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    assert len(numeric_features) == 26, "Numeric feature count mismatch!"
    assert len(categorical_features) == 7, "Categorical feature count mismatch!"

    # Numerical Transformer:
    # 1. Median Imputation: Fills any potential missing values with the median.
    #    Median is preferred over mean because it is robust to extreme outliers (e.g. high MonthlyIncome).
    # 2. Standard Scaling: Centers features to mean=0 and variance=1.
    #    Critical for Logistic Regression because unscaled features with large ranges
    #    (e.g., MonthlyRate in thousands) would dominate the regularization penalty.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Transformer:
    # 1. Most-Frequent Imputation: Replaces missing values with the mode of that column.
    # 2. One-Hot Encoding: Converts nominal text strings into binary indicator columns.
    #    Setting `handle_unknown="ignore"` ensures the pipeline will not throw an error
    #    if an unseen category appears during inference; it will simply assign all zeros.
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine transformers into a single ColumnTransformer.
    # Important: This preprocessor is NOT fitted globally. It is embedded into model
    # pipelines and fitted strictly on X_train during cross-validation, preventing data leakage.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 5-fold Stratified Cross-Validation Strategy for hyperparameter tuning and model selection
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------------------------------
    # 5. Model Training & Cross-Validation on X_train ONLY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: CROSS-VALIDATION ON TRAINING PARTITION ONLY")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 5. Model Training & Cross-Validation on X_train ONLY
    # -------------------------------------------------------------------------
    # In strict machine learning methodology, we never look at the test set
    # while tuning or comparing models. All training and hyperparameter search
    # must take place exclusively on the training set (X_train, y_train).
    #
    # We evaluate three distinct algorithm families:
    # 1. Decision Trees: Fast, highly interpretable, but prone to high-variance overfitting.
    # 2. Random Forests: Bagging ensemble of randomized trees; reduces variance but slower.
    # 3. Logistic Regression: Linear model providing smooth, well-behaved probabilities.

    # -------------------------------------------------------------------------
    # A. Baseline Decision Tree
    # -------------------------------------------------------------------------
    # We use class_weight="balanced" so the tree penalizes errors on the minority
    # attrition class inversely proportional to class frequencies.
    dt_base = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ])
    dt_base_cv = cross_validate(dt_base, X_train, y_train, cv=cv_strategy, scoring=["f1", "roc_auc"], n_jobs=-1)
    dt_base_f1 = dt_base_cv["test_f1"].mean()
    dt_base_auc = dt_base_cv["test_roc_auc"].mean()
    print(f"1. Baseline Decision Tree    - CV F1 (Yes): {dt_base_f1:.4f}, CV ROC-AUC: {dt_base_auc:.4f}")

    # -------------------------------------------------------------------------
    # B. Tuned Decision Tree via GridSearchCV
    # -------------------------------------------------------------------------
    # Decision trees overfit when allowed to grow indefinitely. By tuning max_depth,
    # min_samples_split, and min_samples_leaf, we restrict tree complexity (pruning).
    dt_tuning_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ])
    dt_grid = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [3, 5, 7, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8]
    }
    # GridSearchCV explores every combination in dt_grid using 5-fold Stratified CV,
    # selecting the parameter set with the highest average validation F1-score.
    dt_search = GridSearchCV(dt_tuning_pipe, dt_grid, scoring="f1", cv=cv_strategy, n_jobs=-1, refit=True)
    dt_search.fit(X_train, y_train)
    best_dt = dt_search.best_estimator_
    dt_tuned_cv = cross_validate(best_dt, X_train, y_train, cv=cv_strategy, scoring=["f1", "roc_auc"], n_jobs=-1)
    dt_tuned_f1 = dt_tuned_cv["test_f1"].mean()
    dt_tuned_auc = dt_tuned_cv["test_roc_auc"].mean()
    print(f"2. Tuned Decision Tree       - CV F1 (Yes): {dt_tuned_f1:.4f}, CV ROC-AUC: {dt_tuned_auc:.4f}")
    print(f"   Best params: {dt_search.best_params_}")

    # -------------------------------------------------------------------------
    # C. Baseline Random Forest
    # -------------------------------------------------------------------------
    # Default 100-tree ensemble with balanced class weighting.
    rf_base = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1))
    ])
    rf_base_cv = cross_validate(rf_base, X_train, y_train, cv=cv_strategy, scoring=["f1", "roc_auc"], n_jobs=-1)
    rf_base_f1 = rf_base_cv["test_f1"].mean()
    rf_base_auc = rf_base_cv["test_roc_auc"].mean()
    print(f"3. Baseline Random Forest    - CV F1 (Yes): {rf_base_f1:.4f}, CV ROC-AUC: {rf_base_auc:.4f}")

    # -------------------------------------------------------------------------
    # D. Tuned Random Forest via GridSearchCV
    # -------------------------------------------------------------------------
    # Tuned over tree count (n_estimators), tree depth (max_depth), leaf size,
    # and feature subsampling (max_features='sqrt' vs 0.5 of all features).
    rf_tuning_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1))
    ])
    rf_grid = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.5]
    }
    rf_search = GridSearchCV(rf_tuning_pipe, rf_grid, scoring="f1", cv=cv_strategy, n_jobs=-1, refit=True)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    rf_tuned_cv = cross_validate(best_rf, X_train, y_train, cv=cv_strategy, scoring=["f1", "roc_auc"], n_jobs=-1)
    rf_tuned_f1 = rf_tuned_cv["test_f1"].mean()
    rf_tuned_auc = rf_tuned_cv["test_roc_auc"].mean()
    print(f"4. Tuned Random Forest       - CV F1 (Yes): {rf_tuned_f1:.4f}, CV ROC-AUC: {rf_tuned_auc:.4f}")
    print(f"   Best params: {rf_search.best_params_}")

    # -------------------------------------------------------------------------
    # E. Baseline Logistic Regression
    # -------------------------------------------------------------------------
    # Standard L2-regularized logistic regression with class_weight='balanced'.
    # max_iter=5000 ensures numerical convergence of the optimization solver.
    lr_base = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))
    ])
    lr_base_cv = cross_validate(lr_base, X_train, y_train, cv=cv_strategy, scoring=["f1", "roc_auc"], n_jobs=-1)
    lr_base_f1 = lr_base_cv["test_f1"].mean()
    lr_base_auc = lr_base_cv["test_roc_auc"].mean()
    print(f"5. Baseline Logistic Reg     - CV F1 (Yes): {lr_base_f1:.4f}, CV ROC-AUC: {lr_base_auc:.4f}")

    # -------------------------------------------------------------------------
    # F. Tuned Logistic Regression via Comprehensive GridSearchCV
    # -------------------------------------------------------------------------
    # We explore:
    # - Regularization strength (C): smaller values = stronger regularization.
    # - Penalty norms (L1 Lasso for sparse feature selection vs L2 Ridge).
    # - Solvers: 'liblinear' (robust for smaller datasets), 'lbfgs', and 'saga'.
    # - Weighting: 'balanced' vs None (standard likelihood).
    lr_tuning_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=10000, random_state=42))
    ])
    lr_param_grid = [
        {
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
            "model__class_weight": ["balanced", None],
            "model__tol": [1e-4, 1e-3]
        },
        {
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
            "model__class_weight": ["balanced", None],
            "model__tol": [1e-4, 1e-3]
        },
        {
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["saga"],
            "model__class_weight": ["balanced", None],
            "model__tol": [1e-4, 1e-3]
        }
    ]
    lr_search = GridSearchCV(lr_tuning_pipe, lr_param_grid, scoring="f1", cv=cv_strategy, n_jobs=-1, refit=True)
    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_
    lr_tuned_cv = cross_validate(best_lr, X_train, y_train, cv=cv_strategy, scoring=["f1", "roc_auc"], n_jobs=-1)
    lr_tuned_f1 = lr_tuned_cv["test_f1"].mean()
    lr_tuned_auc = lr_tuned_cv["test_roc_auc"].mean()
    print(f"6. Tuned Logistic Reg        - CV F1 (Yes): {lr_tuned_f1:.4f}, CV ROC-AUC: {lr_tuned_auc:.4f}")
    print(f"   Best params: {lr_search.best_params_}")

    # Ensure baseline models are fitted on full X_train so they can be evaluated on X_test
    dt_base.fit(X_train, y_train)
    rf_base.fit(X_train, y_train)
    lr_base.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 6. Model Selection Protocol (Conducted STRICTLY Before Testing)
    # -------------------------------------------------------------------------
    # Why F1-Score on the Positive Class ('Attrition = Yes') is the Primary Metric:
    # In an imbalanced dataset where only ~16% leave, an uninformative dummy model
    # that always predicts 'No Attrition' achieves an apparently impressive 84% accuracy,
    # but detects exactly 0 departing employees (Recall = 0, F1 = 0).
    # F1-score calculates the harmonic mean of Precision and Recall on the positive
    # minority class, penalizing models that fail to identify departures or generate
    # excessive false alarms.
    #
    # Secondary Metric: ROC-AUC serves as a tie-breaker if top candidate CV F1 scores
    # differ by less than 0.01.
    print("\n" + "=" * 70)
    print("STEP 6: MODEL SELECTION ON CROSS-VALIDATION RESULTS")
    print("=" * 70)

    cv_results = [
        {"Model": "Decision Tree (Baseline)", "CV_F1": dt_base_f1, "CV_ROC_AUC": dt_base_auc, "Pipeline": dt_base},
        {"Model": "Decision Tree (Tuned)", "CV_F1": dt_tuned_f1, "CV_ROC_AUC": dt_tuned_auc, "Pipeline": best_dt},
        {"Model": "Random Forest (Baseline)", "CV_F1": rf_base_f1, "CV_ROC_AUC": rf_base_auc, "Pipeline": rf_base},
        {"Model": "Random Forest (Tuned)", "CV_F1": rf_tuned_f1, "CV_ROC_AUC": rf_tuned_auc, "Pipeline": best_rf},
        {"Model": "Logistic Regression (Baseline)", "CV_F1": lr_base_f1, "CV_ROC_AUC": lr_base_auc, "Pipeline": lr_base},
        {"Model": "Logistic Regression (Tuned)", "CV_F1": lr_tuned_f1, "CV_ROC_AUC": lr_tuned_auc, "Pipeline": best_lr},
    ]
    cv_df = pd.DataFrame(cv_results).sort_values(by="CV_F1", ascending=False).reset_index(drop=True)
    print(cv_df[["Model", "CV_F1", "CV_ROC_AUC"]].to_string(index=False))

    # Apply tie-breaker rule deterministically
    top1 = cv_df.iloc[0]
    top2 = cv_df.iloc[1]
    if abs(top1["CV_F1"] - top2["CV_F1"]) < 0.01:
        print(f"\nNote: Top two models F1 difference is {abs(top1['CV_F1'] - top2['CV_F1']):.4f} (< 0.01). Consulting ROC-AUC tie breaker.")
        if top2["CV_ROC_AUC"] > top1["CV_ROC_AUC"]:
            selected_row = top2
        else:
            selected_row = top1
    else:
        selected_row = top1

    selected_model_name = selected_row["Model"]
    selected_pipeline = selected_row["Pipeline"]
    print(f"\n>>> SELECTED FINAL MODEL (Pre-Test): {selected_model_name} <<<")
    print(f"    Selected CV F1 (Yes): {selected_row['CV_F1']:.4f}, CV ROC-AUC: {selected_row['CV_ROC_AUC']:.4f}")

    # -------------------------------------------------------------------------
    # 7. Held-Out Test Set Evaluation (294 Samples)
    # -------------------------------------------------------------------------
    # Now that the winning architecture has been objectively selected, we evaluate
    # all models on the held-out 20% test partition (X_test, y_test) to provide
    # an unbiased estimate of generalization performance.
    print("\n" + "=" * 70)
    print("STEP 7: HELD-OUT TEST EVALUATION (294 SAMPLES)")
    print("=" * 70)

    all_models = [
        ("Decision Tree (Baseline)", dt_base),
        ("Decision Tree (Tuned)", best_dt),
        ("Random Forest (Baseline)", rf_base),
        ("Random Forest (Tuned)", best_rf),
        ("Logistic Regression (Baseline)", lr_base),
        ("Logistic Regression (Tuned)", best_lr)
    ]

    test_metrics_list = []
    roc_curves_data = {}

    for name, pipe in all_models:
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        y_test_proba = pipe.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        # zero_division=0 prevents warnings if a model predicts 0 positive cases
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_test_proba)

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_curves_data[name] = (fpr, tpr, auc)

        test_metrics_list.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Precision (Yes)": round(prec, 4),
            "Recall (Yes)": round(rec, 4),
            "F1-Score (Yes)": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

    test_summary_df = pd.DataFrame(test_metrics_list).sort_values(by="F1-Score (Yes)", ascending=False).reset_index(drop=True)
    print("\nFINAL MODEL COMPARISON TABLE (Held-Out Test Set):")
    print(test_summary_df.to_string(index=False))

    # Detailed evaluation for the selected winning model
    print(f"\n--- DETAILED EVALUATION FOR SELECTED MODEL: {selected_model_name} ---")
    y_sel_pred = selected_pipeline.predict(X_test)
    y_sel_proba = selected_pipeline.predict_proba(X_test)[:, 1]

    # Confusion Matrix Breakdown:
    # - True Negative (TN): Employee stayed, model predicted stay. (Correct)
    # - False Positive (FP): Employee stayed, model predicted leave. (False alarm)
    # - False Negative (FN): Employee left, model predicted stay. (Missed departure)
    # - True Positive (TP): Employee left, model predicted leave. (Correct detection)
    cm = confusion_matrix(y_test, y_sel_pred)
    print("Confusion Matrix:")
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],\n [FN={cm[1,0]}, TP={cm[1,1]}]]")

    cr = classification_report(y_test, y_sel_pred, target_names=["No Attrition (0)", "Attrition (1)"], digits=4)
    print("\nClassification Report:")
    print(cr)

    # -------------------------------------------------------------------------
    # 8. Plot & Save High-Resolution Visualizations
    # -------------------------------------------------------------------------
    # Plot 1: Multi-Model ROC Curves
    # Receiver Operating Characteristic curves display the trade-off between
    # sensitivity (True Positive Rate) and 1 - specificity (False Positive Rate)
    # across every possible classification threshold.
    plt.figure(figsize=(10, 7))
    for name, (fpr, tpr, auc) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison (80:20 Split Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_80_20.png", dpi=300)
    plt.close()
    print("\nSaved ROC Curves: roc_curves_80_20.png")

    # Plot 2: Random Forest Top 10 Feature Importances
    # Important Scientific Caution:
    # Gini feature importance reflects predictive correlation in the dataset,
    # NOT direct causal mechanisms. High importance indicates that splitting on this
    # feature effectively separates stayers from leavers, but does not prove that
    # artificially changing this feature will change employee departure decisions.
    rf_feature_names = best_rf.named_steps["preprocessor"].get_feature_names_out()
    rf_importances = best_rf.named_steps["model"].feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": rf_feature_names,
        "Importance": rf_importances
    }).sort_values(by="Importance", ascending=False).head(10).reset_index(drop=True)

    print("\nTop 10 Random Forest Features:")
    print(feat_imp_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1], color="#EA580C")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances - Tuned Random Forest (80:20 Split)")
    plt.tight_layout()
    plt.savefig("rf_feature_importance_80_20.png", dpi=300)
    plt.close()
    print("Saved Feature Importances: rf_feature_importance_80_20.png")

    # -------------------------------------------------------------------------
    # 9. Export Winning Pipeline Artifact
    # -------------------------------------------------------------------------
    # We serialize the entire Scikit-Learn pipeline (ColumnTransformer + Tuned Classifier)
    # using Joblib so the Streamlit application can load and execute predictions natively.
    export_filename = "employee_attrition_model_80_20.joblib"
    joblib.dump(selected_pipeline, export_filename)
    print(f"\nSuccessfully exported pipeline to: {export_filename}")

    # -------------------------------------------------------------------------
    # 10. Native Deserialization & Prediction Parity Check
    # -------------------------------------------------------------------------
    # Defensive Engineering Validation:
    # We immediately reload the saved artifact from disk and verify that:
    # 1. Predictions match the in-memory pipeline exactly (bitwise parity).
    # 2. Probability estimates match within numerical floating-point tolerances (np.allclose).
    print("\n" + "=" * 70)
    print("STEP 10: NATIVE VERIFICATION & PARITY CHECK")
    print("=" * 70)

    reloaded_pipeline = joblib.load(export_filename)
    reloaded_pred = reloaded_pipeline.predict(X_test)
    reloaded_proba = reloaded_pipeline.predict_proba(X_test)

    pred_parity = np.array_equal(y_sel_pred, reloaded_pred)
    proba_parity = np.allclose(selected_pipeline.predict_proba(X_test), reloaded_proba)

    print(f"Reloaded pipeline type: {type(reloaded_pipeline)}")
    print(f"Steps: {[s[0] for s in reloaded_pipeline.steps]}")
    print(f"Prediction parity on test set: {pred_parity}")
    print(f"Probability parity on test set: {proba_parity}")
    print(f"Pipeline classes: {reloaded_pipeline.classes_}")

    assert pred_parity, "Prediction parity check failed!"
    assert proba_parity, "Probability parity check failed!"
    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY WITHOUT WARNINGS OR SHIMS <<<")

    # -------------------------------------------------------------------------
    # 11. Save Summary Results JSON for Auditing & Documentation
    # -------------------------------------------------------------------------
    results_dict = {
        "selected_model": selected_model_name,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_attrition_rate": float(y_train.mean()),
        "test_attrition_rate": float(y_test.mean()),
        "selected_cv_f1": float(selected_row["CV_F1"]),
        "selected_cv_roc_auc": float(selected_row["CV_ROC_AUC"]),
        "test_summary": test_summary_df.to_dict(orient="records"),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "rf_top10": feat_imp_df.to_dict(orient="records")
    }
    with open("retraining_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("Saved metrics summary: retraining_results.json")

if __name__ == "__main__":
    run_retraining()
