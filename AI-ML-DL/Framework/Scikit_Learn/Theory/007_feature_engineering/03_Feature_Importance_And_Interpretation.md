# Feature Importance and Interpretation

---

## Table of Contents

- [Overview](#overview)
- [Tree-Based Feature Importance](#tree-based-feature-importance)
- [Permutation Importance](#permutation-importance)
- [Coefficient-Based Importance](#coefficient-based-importance)
- [Partial Dependence](#partial-dependence)
- [SHAP and LIME](#shap-and-lime)
- [Comparing Importance Methods](#comparing-importance-methods)
- [Best Practices](#best-practices)

---

## Overview

**Feature importance** quantifies the contribution of each feature to model predictions. Different methods suit different model types and interpretability needs. **Interpretation** extends to understanding individual predictions (local) or feature effects (global).

| Method | Model Type | Scope |
|--------|------------|-------|
| **feature_importances_** | Trees, forests | Global |
| **coef_** | Linear models | Global |
| **permutation_importance** | Any model | Global |
| **partial_dependence** | Any model | Global (marginal effect) |
| **SHAP, LIME** | Any model | Local and global |

---

## Tree-Based Feature Importance

Tree-based models (DecisionTree, RandomForest, GradientBoosting, etc.) provide **feature_importances_**. For trees: based on impurity decrease (Gini or MSE). For forests: mean decrease across trees.

### Key Attributes

- **feature_importances_**: Array of importance values (sum to 1 for trees)

### Usage

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

### Interpretation

- Higher value = more important for predictions
- Values are relative (sum to 1)
- Can be biased toward high-cardinality or many-split features

---

## Permutation Importance

**permutation_importance** measures importance by shuffling each feature and observing the drop in score. Model-agnostic and robust.

### Usage

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
print(result.importances_mean)
print(result.importances_std)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_repeats** | Number of shuffle iterations |
| **scoring** | Metric (default: score method) |
| **random_state** | Reproducibility |

### When to Use

- Any model type
- More reliable than tree importance for correlated features
- Use **X_test** to avoid overfitting to training set

---

## Coefficient-Based Importance

For linear models, **coef_** indicates direction and magnitude of each feature's effect. Scale features before fitting so coefficients are comparable.

### Usage

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_scaled, y)
print(model.coef_)
```

### Interpretation

- Magnitude: strength of effect
- Sign: positive or negative relationship
- For L1: zero coef = feature not selected

---

## Partial Dependence

**partial_dependence** and **PartialDependenceDisplay** show the marginal effect of one or two features on predictions, averaging over other features.

### Usage

```python
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay

pdp = partial_dependence(model, X, features=[0, 1], kind="average")
PartialDependenceDisplay.from_estimator(model, X, features=[0, 1])
```

### Parameters

- **features**: Feature indices or names
- **kind**: 'average' (default) or 'individual'

### Interpretation

- Flat line: feature has little effect
- Non-linear: complex relationship
- ICE (individual) plots show per-sample variation

---

## SHAP and LIME

**SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) provide local explanations. Not in scikit-learn core; use **shap** and **lime** packages.

### SHAP (summary)

```python
import shap

explainer = shap.TreeExplainer(model, X)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

### LIME (summary)

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=...)
exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
```

---

## Comparing Importance Methods

| Method | Pros | Cons |
|--------|------|------|
| **feature_importances_** | Fast, built-in | Tree models only; can be biased |
| **permutation_importance** | Model-agnostic, robust | Slower |
| **coef_** | Direct, interpretable | Linear only |
| **partial_dependence** | Shows marginal effect | Expensive for many features |
| **SHAP** | Theoretically grounded | Slower; external package |

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use **permutation_importance** for model-agnostic view | Robust, comparable across models |
| Use **X_test** for permutation importance | Avoid overfitting |
| **Scale** features before linear models | Comparable coef_ |
| Use **PartialDependenceDisplay** for visualization | Clear marginal effects |
| Combine multiple methods | Cross-validate interpretation |
