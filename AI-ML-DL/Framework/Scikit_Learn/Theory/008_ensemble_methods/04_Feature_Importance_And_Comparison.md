# Feature Importance and Ensemble Comparison

---

## Table of Contents

- [Overview](#overview)
- [Feature Importance in Tree Ensembles](#feature-importance-in-tree-ensembles)
- [Comparing Importance Across Methods](#comparing-importance-across-methods)
- [Partial Dependence Plots](#partial-dependence-plots)
- [Ensemble Comparison Workflow](#ensemble-comparison-workflow)
- [Choosing an Ensemble Method](#choosing-an-ensemble-method)

---

## Overview

Tree-based ensembles provide **feature importance** and support **partial dependence** analysis for interpretability. Comparing multiple ensemble methods on a single dataset helps select the best model for a given task.

---

## Feature Importance in Tree Ensembles

### How It Is Computed

For tree-based models, importance is typically based on **impurity reduction** (Gini or MSE) at splits involving each feature, summed over all nodes and trees, then normalized.

**Formula** (simplified): For each feature, sum the weighted impurity decrease at every split on that feature, where the weight is the number of samples reaching the node.

### Models with feature_importances_

| Model | Importance Type |
|-------|-----------------|
| **RandomForestClassifier/Regressor** | Mean decrease in impurity |
| **GradientBoostingClassifier/Regressor** | Same |
| **AdaBoostClassifier/Regressor** | Weighted impurity decrease |
| **ExtraTreesClassifier/Regressor** | Same as Random Forest |

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
print(rf.feature_importances_.sum())
```

### Interpretation

- **Higher value**: Feature contributes more to predictions
- **Zero**: Feature never used in any split
- **Sum**: Typically 1.0 for tree ensembles

---

## Comparing Importance Across Methods

Different ensembles may rank features differently due to:

- **Algorithm**: Bagging vs boosting
- **Randomization**: max_features, bootstrap
- **Loss function**: Gini vs entropy vs MSE

### Workflow

1. Fit multiple ensemble models
2. Extract **feature_importances_** from each
3. Compare rankings or aggregate (e.g., mean, median)
4. Use consensus for feature selection

```python
importances = {}
for name, model in [("RF", rf), ("GB", gb), ("Ada", ada)]:
    model.fit(X_train, y_train)
    importances[name] = model.feature_importances_

mean_importance = np.mean(list(importances.values()), axis=0)
top_features = np.argsort(mean_importance)[-5:][::-1]
```

### Correlation of Importance

Compute correlation between importance vectors to see agreement:

```python
corr = np.corrcoef(rf.feature_importances_, gb.feature_importances_)[0, 1]
```

---

## Partial Dependence Plots

**Partial dependence** shows the average effect of one or more features on the predicted outcome, marginalizing over other features.

### Definition

For feature set \(S\), partial dependence is:

\[ \text{PD}_S(x_S) = \mathbb{E}_{X_C}[f(x_S, X_C)] \]

where \(X_C\) are the other features. Estimated by varying \(x_S\) and averaging predictions over the training data.

### PartialDependenceDisplay

**PartialDependenceDisplay** creates one-way or two-way partial dependence plots.

```python
from sklearn.inspection import PartialDependenceDisplay

disp = PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 2], kind="average"
)
disp.plot()
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| **features** | Feature indices: single, list, or list of tuples for interactions |
| **kind** | `"average"` (PDP) or `"individual"` (ICE curves) |
| **grid_resolution** | Number of grid points per feature |

### One-Way vs Two-Way

- **One-way**: `features=[0, 2]` — separate curves for features 0 and 2
- **Two-way**: `features=[(0, 2)]` — heatmap of interaction between features 0 and 2

### Individual Conditional Expectation (ICE)

**kind="individual"** plots one curve per sample, showing how each prediction changes with the feature. Useful for detecting heterogeneity.

```python
disp = PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0], kind="individual"
)
```

### partial_dependence Function

For programmatic access to values:

```python
from sklearn.inspection import partial_dependence

pd_values, grid = partial_dependence(model, X_train, features=[0])
```

---

## Ensemble Comparison Workflow

### Steps

1. **Prepare data**: Train/test split, optional scaling
2. **Define models**: Multiple ensemble (and non-ensemble) models
3. **Fit and evaluate**: Accuracy/R2, F1, MSE, etc.
4. **Cross-validate**: Use **cross_val_score** for robust comparison
5. **Compare**: Summarize metrics, training time, interpretability

```python
from sklearn.model_selection import cross_val_score

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Voting": VotingClassifier(estimators=[...], voting="soft"),
    "Stacking": StackingClassifier(estimators=[...], cv=5),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### Metrics to Compare

| Task | Metrics |
|------|---------|
| **Classification** | Accuracy, F1, ROC-AUC, precision, recall |
| **Regression** | MSE, RMSE, MAE, R2 |

### Training Time

Use **time.perf_counter()** to compare fit times. HistGradientBoosting and Extra Trees are typically faster than GradientBoosting and Random Forest.

---

## Choosing an Ensemble Method

### Decision Guide

| Criterion | Recommendation |
|-----------|----------------|
| **Default choice** | Random Forest |
| **Best accuracy (tabular)** | Gradient Boosting or HistGradientBoosting |
| **Large datasets** | HistGradientBoosting |
| **Missing values** | HistGradientBoosting |
| **Categorical features** | HistGradientBoosting |
| **Speed** | Extra Trees, HistGradientBoosting |
| **Interpretability** | Random Forest (feature_importances_, PDP) |
| **Heterogeneous base models** | Voting or Stacking |

### Trade-offs

| Method | Pros | Cons |
|--------|------|------|
| **Random Forest** | Robust, parallel, OOB | Can be slow on large data |
| **Gradient Boosting** | Often best accuracy | Slow, sequential |
| **HistGradientBoosting** | Fast, missing/categorical | Less tunable than GB |
| **AdaBoost** | Simple, works with weak learners | Sensitive to noise |
| **Voting** | Simple, fast | Fixed combination rule |
| **Stacking** | Learned combination | Overfitting risk, slower |

---

## Summary

- **feature_importances_** in tree ensembles reflect impurity-based contribution
- Compare importance across models to find stable, important features
- **PartialDependenceDisplay** and **partial_dependence** support model interpretation
- Use **cross_val_score** and multiple metrics for fair ensemble comparison
- Choose ensemble by data size, missing values, categoricals, and accuracy vs speed
