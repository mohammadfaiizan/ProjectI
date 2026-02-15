# MLOps and Interpretation

---

## Table of Contents

- [Overview](#overview)
- [MLflow Integration](#mlflow-integration)
- [autolog and log_model](#autolog-and-log_model)
- [Experiment Tracking Patterns](#experiment-tracking-patterns)
- [SHAP for Model Interpretation](#shap-for-model-interpretation)
- [LIME for Local Interpretation](#lime-for-local-interpretation)
- [Testing ML Models](#testing-ml-models)
- [parametrize_with_checks](#parametrize_with_checks)
- [Data-Driven Tests](#data-driven-tests)
- [Experiment Tracking](#experiment-tracking)
- [Best Practices](#best-practices)

---

## Overview

This document covers **MLOps** (MLflow, experiment tracking), **model interpretation** (SHAP, LIME), and **testing** strategies for scikit-learn models. These practices ensure reproducibility, interpretability, and reliability in production.

---

## MLflow Integration

### What is MLflow

**MLflow** is an open-source platform for managing the ML lifecycle: experiments, runs, models, and deployment. Scikit-learn integrates via **mlflow.sklearn**.

### Installation

```bash
pip install mlflow
```

### Basic Workflow

1. Start a run with `mlflow.start_run()`
2. Log parameters, metrics, and artifacts
3. Log the model with `mlflow.sklearn.log_model()`
4. Register models in the Model Registry for deployment

---

## autolog and log_model

### mlflow.sklearn.autolog

**autolog** automatically logs parameters, metrics, and artifacts when fitting sklearn estimators:

```python
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

with mlflow.start_run():
    clf = LogisticRegression(C=0.1)
    clf.fit(X_train, y_train)
    # Parameters (C, max_iter, etc.) and metrics (accuracy) logged automatically
    mlflow.sklearn.log_model(clf, "model")
```

### log_model

Log a fitted model as an artifact:

```python
mlflow.sklearn.log_model(clf, "model")
# Model saved as artifact; can be loaded with mlflow.sklearn.load_model()
```

### log_metric and log_param

Manual logging for custom metrics and parameters:

```python
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("cv_accuracy", cv_scores.mean())
mlflow.log_metrics({"precision": 0.9, "recall": 0.85})
```

---

## Experiment Tracking Patterns

### Run Structure

- **Experiment**: A named container for runs (e.g., "iris_classification")
- **Run**: A single execution with params, metrics, artifacts
- **Artifacts**: Model files, plots, configs

### Nested Runs

Use `mlflow.start_run(nested=True)` for hyperparameter search or cross-validation logging.

### Manual Tracking Without MLflow

For lightweight setups, use JSON files or a simple database:

```python
import json
from pathlib import Path

run = {
    "config": {"model": "RF", "n_estimators": 50},
    "metrics": {"accuracy": 0.95, "cv_mean": 0.93}
}
Path("runs").mkdir(exist_ok=True)
with open("runs/run_001.json", "w") as f:
    json.dump(run, f, indent=2)
```

---

## SHAP for Model Interpretation

### What is SHAP

**SHAP** (SHapley Additive exPlanations) provides feature importance values that sum to the difference between the model output and the base (expected) value. It is model-agnostic but has optimized implementations for tree models.

### TreeExplainer

For tree-based models (RandomForest, GradientBoosting, XGBoost), use **TreeExplainer**:

```python
import shap

explainer = shap.TreeExplainer(clf, X_train)
shap_values = explainer.shap_values(X_test)
# For binary: (n_samples, n_features)
# For multiclass: list of arrays, one per class
```

### Summary Plot

```python
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Force Plot

For single-instance explanation:

```python
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
```

### Compatibility

- **TreeExplainer**: RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost
- **KernelExplainer**: Any model (slower, sampling-based)
- **LinearExplainer**: Linear models

---

## LIME for Local Interpretation

### What is LIME

**LIME** (Local Interpretable Model-agnostic Explanations) fits a simple linear model locally around a prediction to explain it. The linear coefficients are the local feature importance.

### LimeTabularExplainer

```python
import lime
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train, feature_names=feature_names,
    class_names=["Class0", "Class1"], mode="classification"
)
exp = explainer.explain_instance(X_test[0], clf.predict_proba, num_features=5)
exp.show_in_notebook()
```

### Concepts

- **Perturbation**: Generate synthetic samples near the instance
- **Weighting**: Weight samples by distance to the instance
- **Interpretable model**: Fit weighted linear regression to (perturbed_X, model(perturbed_X))
- **Coefficients**: Local feature importance

### When to Use

- **SHAP**: Global and local, theoretically grounded; use TreeExplainer for trees
- **LIME**: Local only; model-agnostic; good for non-tree models

---

## Testing ML Models

### Why Test ML Code

- **API stability**: Estimators follow fit/predict contract
- **Reproducibility**: Same input yields same output with random_state
- **Regression**: Catch unintended behavior changes
- **Integration**: Pipelines and cross-validation work correctly

### Unit Tests

Test individual components: transformers, estimators, metrics. Use **assert** for expected shapes, value ranges, and invariants.

---

## parametrize_with_checks

### Purpose

**parametrize_with_checks** runs sklearn's built-in **estimator checks** on a list of estimators. These checks verify API compliance, input validation, and idempotency.

### Usage

```python
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@parametrize_with_checks([LogisticRegression(), RandomForestClassifier()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)
```

### Common Checks

| Check | Purpose |
|-------|---------|
| **check_estimator** | Full estimator contract |
| **check_fit_score_takes_y** | Supervised estimators accept y |
| **check_no_attributes_set_in_init** | Attributes set in fit, not init |
| **check_estimators_dtypes** | Consistent dtypes |
| **check_fit_idempotent** | fit(X,y); fit(X,y) same as fit(X,y) |

---

## Data-Driven Tests

### Pattern

Generate data, fit, predict, assert invariants:

```python
def test_classifier_output_shape():
    X, y = make_classification(n_samples=100)
    clf = LogisticRegression().fit(X, y)
    pred = clf.predict(X)
    assert pred.shape[0] == X.shape[0]
    assert set(pred).issubset(set(clf.classes_))
```

### Input Validation Tests

```python
from sklearn.utils.validation import check_array

def test_rejects_nan():
    X = np.array([[1, 2], [3, np.nan]])
    with pytest.raises(ValueError):
        check_array(X, allow_nan=False)
```

### Regression Tests

Save expected outputs for critical paths and compare. Update when behavior intentionally changes.

---

## Experiment Tracking

### What to Log

| Item | Example |
|------|---------|
| **Parameters** | n_estimators, max_depth, C |
| **Metrics** | accuracy, precision, recall, AUC |
| **Artifacts** | model file, config, plots |
| **Environment** | sklearn version, Python version |
| **Data** | hash or path to dataset |

### Reproducibility

- Set **random_state** on all estimators and CV splitters
- Log **sklearn.__version__**
- Use fixed train/test split or seed for data loading

---

## Best Practices

| Practice | Recommendation |
|----------|----------------|
| **MLflow** | Use autolog for quick experiments; manual log for production |
| **SHAP** | Prefer TreeExplainer for trees; KernelExplainer for others |
| **LIME** | Use for local explanations when SHAP is slow |
| **Testing** | Use parametrize_with_checks for custom estimators |
| **Tracking** | Log params, metrics, and model version |
| **Reproducibility** | Always set random_state and log versions |

---

## Summary

- **MLflow**: autolog, log_model, log_metric for experiment tracking
- **SHAP**: TreeExplainer for trees; KernelExplainer for any model
- **LIME**: Local linear approximation for interpretability
- **parametrize_with_checks**: Validate estimator API compliance
- **Data-driven tests**: Assert shapes, classes, and invariants
- **Experiment tracking**: Log config, metrics, artifacts, and environment
