# Employee Attrition Predictor — Machine Learning & Analytics System

An enterprise decision-support machine learning system to analyze, evaluate, and estimate employee attrition risk using the IBM HR Analytics dataset (1,470 records, 35 features). Built with Scikit-Learn pipelines, stratified validation, and an interactive Streamlit console inspired by enterprise design standards (Amazon and Uber internal consoles).

---

## 1. Project Overview & Business Problem

Employee turnover creates substantial organizational costs, including loss of institutional knowledge, recruitment overhead, productivity deficits, and team disruption. The objective of this project is to:
- Conduct exploratory data analysis to identify significant correlates of workforce turnover.
- Build reproducible machine learning classification pipelines incorporating missing value imputation, scaling, categorical encoding, and feature-engineered tenure ratios without data leakage.
- Compare candidate models (Decision Trees, Random Forests, Logistic Regression) using rigorous cross-validation and pre-test selection protocols.
- Deliver an interactive, local decision-support console for workforce analytics that embeds responsible AI principles and strictly forbids automated personnel actions.

---

## 2. Dataset Description

- **Source**: IBM HR Analytics Employee Attrition & Performance Dataset.
- **Observations**: 1,470 employee records.
- **Raw Features**: 35 columns (demographics, compensation, tenure, job roles, satisfaction metrics).
- **Target Variable**: `Attrition` (`'Yes'` = Employee departed, `'No'` = Employee retained).
- **Class Balance**: 1,233 No (83.88%), 237 Yes (16.12%) — **moderately imbalanced**.
- **Dropped Features (4 zero-variance / identifier columns)**:
  - `EmployeeNumber`: Unique numerical identifier (non-generalizable).
  - `EmployeeCount`: Constant value `1` across all records.
  - `Over18`: Constant value `'Y'` across all records.
  - `StandardHours`: Constant value `80` across all records.
- **Model Input Features**: **33 total features** (26 numerical + 7 categorical).

---

## 3. Environment Setup & Execution

The project runs in a dedicated, isolated virtual environment configured with **Python 3.11.17** and **Scikit-Learn 1.6.1** to ensure native artifact loading with zero unpickling shims or warnings.

### A. Environment Initialization
If recreating the environment from scratch using `uv`:
```bash
# Create isolated Python 3.11 virtual environment
uv venv .venv --python 3.11

# Install exact pinned dependencies
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Alternatively, using standard Python:
```bash
# Activate existing local venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### B. Launching the Interactive Streamlit Application
Execute from the repository root directory:
```bash
# Using the dedicated virtual environment executable:
.venv\Scripts\streamlit.exe run app.py

# Or if the virtual environment is activated:
streamlit run app.py
```
The application will launch locally at `http://127.0.0.1:8501`.

### C. Running Automated Tests
Execute the full test suite (pipeline validation, API backend endpoints, and frontend mock client):
```bash
.venv\Scripts\python.exe -m pytest tests -v
```

### D. Decoupled Architecture Execution (FastAPI Backend + Streamlit Frontend)
To run the decoupled client-server architecture locally in separate processes:

1. **Terminal 1 — Start FastAPI Backend**:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
   - Interactive OpenAPI documentation: `http://127.0.0.1:8000/docs`
   - Health probe endpoint: `http://127.0.0.1:8000/health`

2. **Terminal 2 — Start Decoupled Streamlit Frontend**:
   ```bash
   # Windows PowerShell:
   $env:BACKEND_API_URL="http://127.0.0.1:8000"
   .\.venv\Scripts\streamlit.exe run frontend/app.py

   # Windows Command Prompt (cmd):
   set BACKEND_API_URL=http://127.0.0.1:8000
   .\.venv\Scripts\streamlit.exe run frontend/app.py
   ```
   - Frontend console: `http://127.0.0.1:8501`

---


## 4. Machine Learning Pipeline Architecture

### A. 80:20 Stratified Partitioning
To adhere strictly to assignment requirements while preserving the minority target distribution:
- **Total Records**: 1,470
- **Training Set (`X_train`)**: **1,176 records (80.0%)** | 986 No (83.84%), 190 Yes (16.16%)
- **Held-Out Test Set (`X_test`)**: **294 records (20.0%)** | 247 No (84.01%), 47 Yes (15.99%)
- **Stratification Check**: $| \text{Rate}_{\text{train}} - \text{Rate}_{\text{test}} | < 0.2\%$, preserving representative class ratios.

### B. Preprocessing & Feature Engineering
All transformations are encapsulated inside an `sklearn.compose.ColumnTransformer` fitted **exclusively on `X_train`** to prevent data leakage:
- **Numerical Features (26)**: Median imputation via `SimpleImputer(strategy='median')` followed by standardization via `StandardScaler()`.
- **Categorical Features (7)**: Most-frequent imputation via `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
- **Engineered Ratios (3)**: Computed at the row level with mathematical zero-denominator protection:
  $$\text{CompanyExperienceRatio} = \begin{cases} \frac{\text{YearsAtCompany}}{\text{TotalWorkingYears}} & \text{if } \text{TotalWorkingYears} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
  $$\text{CurrentRoleTenureRatio} = \begin{cases} \frac{\text{YearsInCurrentRole}}{\text{YearsAtCompany}} & \text{if } \text{YearsAtCompany} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
  $$\text{ManagerTenureRatio} = \begin{cases} \frac{\text{YearsWithCurrManager}}{\text{YearsAtCompany}} & \text{if } \text{YearsAtCompany} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

### C. Pre-Test Model Selection Protocol
Candidate architectures were trained and cross-validated using **5-Fold Stratified Cross-Validation on the training partition ($N=1,176$) only**:
- **Primary Selection Metric**: Attrition = Yes (Class 1) F1-Score (`scoring='f1'`).
- **Secondary Discriminant**: ROC-AUC (`scoring='roc_auc'`), consulted if candidate F1-scores differ by $< 0.01$.

#### Training Partition 5-Fold Cross-Validation Results:
| Candidate Model | Mean CV F1 (Yes) | Mean CV ROC-AUC | Optimal Hyperparameters |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (Tuned)** | **0.5913** | **0.8440** | `C=1`, `penalty='l2'`, `solver='liblinear'`, `class_weight=None`, `tol=1e-4` |
| Logistic Regression (Baseline) | 0.4916 | 0.8326 | Default (`class_weight='balanced'`) |
| Decision Tree (Tuned) | 0.4029 | 0.7051 | `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=8`, `min_samples_split=20` |
| Random Forest (Tuned) | 0.3869 | 0.8012 | `n_estimators=100`, `max_depth=10`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=2` |
| Decision Tree (Baseline) | 0.3054 | 0.5875 | Default (`class_weight='balanced'`) |
| Random Forest (Baseline) | 0.2717 | 0.7827 | Default (`class_weight='balanced'`) |

**Selection Decision**: Tuned Logistic Regression won decisively with a CV F1 margin of $+0.0997$ over the runner-up, and was selected as the final model prior to touching the test set.

### D. Final Held-Out Test Evaluation ($N=294$)
Evaluated once on the untouched 20% test partition:
| Model | Train Accuracy | Test Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Tuned)** *(Selected)* | **0.9065** | **0.8605** | **0.6154** | **0.3404** | **0.4384** | **0.8163** |
| Decision Tree (Tuned) | 0.8231 | 0.7789 | 0.3816 | 0.6170 | 0.4715 | 0.7183 |
| Logistic Regression (Baseline) | 0.7832 | 0.7585 | 0.3571 | 0.6383 | 0.4580 | 0.8033 |
| Decision Tree (Baseline) | 1.0000 | 0.7721 | 0.2727 | 0.2553 | 0.2637 | 0.5629 |
| Random Forest (Tuned) | 0.9983 | 0.8231 | 0.3529 | 0.1277 | 0.1875 | 0.7800 |
| Random Forest (Baseline) | 1.0000 | 0.8469 | 0.6667 | 0.0851 | 0.1509 | 0.7622 |

#### Detailed Performance for Selected Model:
- **Confusion Matrix**:
  - True Negatives ($TN$): **237** (correctly identified retainers)
  - False Positives ($FP$): **10** (false turnover alarms)
  - False Negatives ($FN$): **31** (missed attrition cases)
  - True Positives ($TP$): **16** (correctly identified departing employees)

---

## 5. Model Limitations & Critical Error Analysis

1. **Recall Deficit on the Positive Class (Recall = 34.04%)**:
   - Out of 47 actual attrition cases in the test set, the model correctly identified **16 employees** while failing to detect **31 employees** (a False Negative rate of $65.96\%$).
   - *Operational Meaning*: HR teams must recognize that a "Lower Model-Estimated Attrition Probability" classification **does not guarantee retention**. Nearly two-thirds of departing employees in the test set did not trigger an elevated alert under the standard 0.50 decision boundary.
2. **False Positives (Precision = 61.54%)**:
   - Out of 26 employees flagged with an elevated departure risk, **10 were false positives** who actually remained.
   - *Operational Meaning*: If automated interventions were tied to model predictions, over a third of targeted employees would receive unnecessary or intrusive attention.
3. **Probability Calibration in Imbalanced Data**:
   - Standard unregularized logistic regression probabilities reflect sample proportions and feature linear combinations. They are **not calibrated Bayesian probabilities of an individual employee's real-world departure**.
   - Output percentages must be treated strictly as **relative diagnostic indicators**, not definitive forecasts.
4. **Historical Observational Bias**:
   - The IBM HR dataset is a synthetic snapshot reflecting historical correlations. Market dynamics, external economic opportunities, family considerations, and team changes not captured in the 35 columns often dictate actual career decisions.

---

## 6. Ethical AI, Fairness & Responsible HR Governance

1. **Strict Prohibition Against Adverse Employment Decisions**:
   - Model predictions must **never** be used as the sole or automated basis for hiring, firing, promotion denial, performance demotion, or compensation changes.
   - Algorithmic profiling in employment violates ethical best practices and increases legal risks under equal opportunity and workforce protection regulations.
2. **Correlation vs. Individual Causation**:
   - Top predictive factors (e.g., lower monthly income, working overtime, low manager tenure ratio) are **statistical correlates**, not proven individual causal drivers.
   - Adjusting a single feature (e.g., offering an ad-hoc raise) does not guarantee an employee will stay, as turnover decisions are multi-factorial.
3. **Protected Demographic Attributes**:
   - The model includes demographic attributes (`Age`, `Gender`, `MaritalStatus`). 
   - Model outputs must be continuously audited across protected demographic classes to prevent disparate impact or discriminatory age profiling.
4. **Human-in-the-Loop Protocol**:
   - An elevated risk indicator should only trigger an optional, supportive, confidential 1-on-1 career check-in between an HR Business Partner and the employee to discuss workload balance, career trajectory, and workplace satisfaction.

---

## 7. Deliverables Inventory

| Category | File Name | Size | Purpose & Role |
| :--- | :--- | :---: | :--- |
| **Runtime Application** | [`app.py`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/app.py) | 22.8 KB | **Core Application**: Enterprise Streamlit user interface collecting 33 features, evaluating pipeline inference, and rendering diagnostic cards. |
| **Runtime Model** | [`employee_attrition_model_80_20.joblib`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/employee_attrition_model_80_20.joblib) | 8.7 KB | **Active Pipeline Artifact**: Trained and verified Scikit-Learn 1.6.1 ColumnTransformer + Tuned Logistic Regression pipeline. |
| **Runtime Config** | [`requirements.txt`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/requirements.txt) | 243 B | **Dependency Specification**: Pinned dependency list matching the verified runtime environment. |
| **Supporting Analysis** | [`Employee_Attrition_Analysis_80_20.ipynb`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/Employee_Attrition_Analysis_80_20.ipynb) | 825 KB | **Assignment Deliverable**: Full Jupyter notebook (39 cells) covering EDA, 5 written insights, 80:20 retraining, tuning, and evaluation. |
| **Supporting Data** | [`WA_Fn-UseC_-HR-Employee-Attrition.csv`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/WA_Fn-UseC_-HR-Employee-Attrition.csv) | 228 KB | **Working Dataset**: Tab-separated HR analytics data (1,470 records). |
| **Supporting Code** | [`train_80_20.py`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/train_80_20.py) | 17.3 KB | **Retraining Script**: Deterministic execution script for cross-validation, tuning, evaluation, and artifact generation. |
| **Supporting Tests** | [`tests/test_pipeline.py`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/tests/test_pipeline.py) | 5.9 KB | **Automated Test Suite**: 9 unit tests verifying schema, ratios, dynamic mapping, and defensive handling. |
| **Supporting Results** | [`retraining_results.json`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/retraining_results.json) | 3.3 KB | **Metrics Audit Log**: JSON export of all training CV scores, test metrics, confusion matrix, and feature importances. |
| **Supporting Visuals** | [`roc_curves_80_20.png`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/roc_curves_80_20.png) | 350 KB | **Evaluation Visual**: High-resolution plot comparing ROC curves of all candidate models on the 80:20 test split. |
| **Supporting Visuals** | [`rf_feature_importance_80_20.png`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/rf_feature_importance_80_20.png) | 144 KB | **Evaluation Visual**: High-resolution bar chart of top 10 Random Forest Gini feature importances. |
| **Legacy Artifact** | [`employee_attrition_model.joblib`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/employee_attrition_model.joblib) | 8.7 KB | **Preserved Historical Model**: Original legacy model (60:40 split) preserved untouched at its original location. |

> **Runtime vs. Supporting Distinction**:
> - To execute the Streamlit application, only **`app.py`**, **`employee_attrition_model_80_20.joblib`**, and the installed virtual environment (**`requirements.txt`**) are required.
> - All other files (`*.ipynb`, `train_80_20.py`, `tests/`, `*.png`, `*.json`) are supporting analysis, verification, and academic submission deliverables.

---

## 8. Codebase Guide (How to Read the Code)

If you are a student, beginner, or new contributor exploring this repository, this guide explains how the project is organized and how data flows through the system.

### A. Recommended Reading Order for Beginners
To build an intuitive understanding of the machine learning lifecycle, explore the files in this sequence:

1. **[`README.md`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/README.md)**: Start here to understand the business problem, dataset structure, model selection protocol, and ethical guardrails.
2. **[`WA_Fn-UseC_-HR-Employee-Attrition.csv`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/WA_Fn-UseC_-HR-Employee-Attrition.csv)**: Inspect the raw tab-separated dataset (1,470 employee records, 35 columns) to understand the input features.
3. **[`Employee_Attrition_Analysis_80_20.ipynb`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/Employee_Attrition_Analysis_80_20.ipynb)**: Walk through the exploratory data analysis (EDA), 8+ visualizations, 5 written analytical findings, feature engineering, and model comparisons with interactive plots and outputs.
4. **[`train_80_20.py`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/train_80_20.py)**: Study the production-style training script to see how data leakage is prevented by fitting transformers strictly on `X_train`, how 5-fold cross-validation is performed, and how models are selected and exported.
5. **[`tests/test_pipeline.py`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/tests/test_pipeline.py)**: Review the automated test suite to understand how machine learning systems are tested defensively (environment pinning, schema completeness, zero-division resilience, and dynamic probability indexing).
6. **[`app.py`](file:///c:/Users/yashu/OneDrive/Desktop/Employee_Attrition_Predictor/app.py)**: Examine how the trained model is loaded into an interactive web interface with cached resource loading, input validation, real-time inference, and responsible AI governance.

### B. Training Flow vs. Inference Flow

Understanding where training happens versus where inference happens is fundamental to machine learning engineering:

```
========================================================================================
TRAINING WORKFLOW (train_80_20.py & Notebook)
========================================================================================
[Raw CSV Dataset]
       │
       ▼
[Feature Engineering] ──> Calculates 3 tenure ratios with zero-denominator safety
       │
       ▼
[Feature Selection]   ──> Drops Attrition (target), EmployeeNumber, and 3 constant columns
       │
       ▼
[Stratified 80:20 Split]
   ├──> Test Partition (20%, N=294): Held out completely untouched
   └──> Training Partition (80%, N=1,176):
            │
            ▼
        [ColumnTransformer] ──> Fits median imputer, standard scaler, and one-hot encoder
            │
            ▼
        [5-Fold Stratified CV & Tuning] ──> Compares Decision Tree, Random Forest, Logistic Reg
            │
            ▼
        [Pre-Test Model Selection] ──> Tuned Logistic Regression wins on CV F1 (0.5913)
            │
            ▼
        [Single Held-Out Test Evaluation] ──> Unbiased benchmark on 294 test samples
            │
            ▼
        [Artifact Serialization] ──> Saves pipeline to employee_attrition_model_80_20.joblib


========================================================================================
INFERENCE WORKFLOW (app.py Web Application)
========================================================================================
[User Web Form (30 raw inputs)]
       │
       ▼
[Consistency Validation] ──> Checks domain constraints (e.g., tenure <= total experience)
       │
       ▼
[Feature Engineering]    ──> Computes 3 tenure ratios using identical zero-safe logic
       │
       ▼
[DataFrame Assembly]     ──> Constructs 1-row DataFrame matching the exact 33-feature schema
       │
       ▼
[Pre-Trained Pipeline]   ──> Loaded natively from employee_attrition_model_80_20.joblib (cached)
       │
       ▼
[Preprocessing & Model]  ──> Transforms features & evaluates logistic sigmoid equation
       │
       ▼
[Dynamic Probability Mapping] ──> Locates positive class (1) dynamically from pipeline.classes_
       │
       ▼
[Diagnostic Presentation] ──> Displays estimated probability %, decision boundary (50%),
                              computed tenure context, and responsible HR governance advisory
========================================================================================
```

### C. Execution Cheat Sheet

All commands should be executed from the project root directory using the isolated virtual environment:

```bash
# 1. Run the interactive Streamlit web dashboard:
.venv\Scripts\streamlit.exe run app.py

# 2. Run the automated QA test suite:
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v

# 3. Re-run model training and evaluation (reproducible):
.venv\Scripts\python.exe train_80_20.py
```
